"""Offline study for the clinical risk console. Real data: NIDDK Pima diabetes (768 patients,
public). Logistic regression + bootstrap ensemble; calibration; OOD leverage; threshold sweep.
Zeros in Glucose/BloodPressure/BMI are known missing-value codes -> those rows are excluded,
stated on the page. Seeded, reproducible."""
import csv, json, math
import numpy as np

rng = np.random.default_rng(7)
rows = list(csv.DictReader(open('diabetes.csv')))
FEATS = ['Glucose','BloodPressure','BMI','Age','DiabetesPedigreeFunction']
LABELS = {'Glucose':'plasma glucose (mg/dL)','BloodPressure':'diastolic blood pressure (mmHg)',
          'BMI':'body mass index','Age':'age (years)','DiabetesPedigreeFunction':'family history score'}
ACTIONABLE = ['Glucose','BloodPressure','BMI']
X, y = [], []
for r in rows:
    v = [float(r[f]) for f in FEATS]
    if v[0]==0 or v[1]==0 or v[2]==0: continue
    X.append(v); y.append(int(r['Outcome']))
X = np.array(X); y = np.array(y)
n, k = X.shape
print(f'n={n} (from 768; zeros-as-missing excluded), positives={y.sum()} ({y.mean():.1%})')

mu, sd = X.mean(0), X.std(0)
Z = (X - mu) / sd

def fit_logistic(Zt, yt, iters=300, lr=0.5, l2=1e-3):
    w = np.zeros(Zt.shape[1]+1)
    A = np.hstack([np.ones((len(Zt),1)), Zt])
    for _ in range(iters):
        p = 1/(1+np.exp(-A@w))
        g = A.T@(p-yt)/len(yt) + l2*np.r_[0, w[1:]]
        w -= lr*g
    return w

w0 = fit_logistic(Z, y)
M = 30
W = np.array([fit_logistic(*(lambda idx: (Z[idx], y[idx]))(rng.integers(0,n,n))) for _ in range(M)])
A = np.hstack([np.ones((n,1)), Z])
P = 1/(1+np.exp(-(A@W.T)))          # n x M
p_mean, p_std = P.mean(1), P.std(1)

# calibration deciles on ensemble mean
qs = np.quantile(p_mean, np.linspace(0,1,11))
cal = []
for i in range(10):
    m = (p_mean>=qs[i]) & (p_mean<=qs[i+1] if i==9 else p_mean<qs[i+1])
    if m.sum()>0: cal.append([float(p_mean[m].mean()), float(y[m].mean()), int(m.sum())])
# discrimination + brier (report honestly)
order = np.argsort(-p_mean)
auc = float(((p_mean[y==1][:,None] > p_mean[y==0][None,:]).mean()))
brier = float(((p_mean-y)**2).mean())
print(f'AUC={auc:.3f}  Brier={brier:.3f}')

# OOD leverage: squared Mahalanobis with diagonal cov (features standardized -> just ||z||^2 scaled)
lev = (Z**2).sum(1)
lev_thr = float(np.quantile(lev, 0.995))
unc_thr = float(np.quantile(p_std, 0.995))
print(f'leverage thr={lev_thr:.1f}, ensemble-std thr={unc_thr:.3f}')

# threshold sweep: clinic capacity story (per 100 patients screened)
sweep=[]
for t in np.linspace(0.15,0.75,25):
    flag = p_mean>=t
    tp = int(((flag)&(y==1)).sum()); fp=int((flag&(y==0)).sum()); fn=int(((~flag)&(y==1)).sum())
    sweep.append([round(float(t),3), round(flag.mean()*100,1), round(tp/max(1,tp+fn),3), round(tp/max(1,tp+fp),3)])
# presets: pick 4 real patients (indices chosen for stories: low-risk, borderline, high-risk, OOD-ish)
i_low  = int(np.argmin(p_mean))
i_high = int(np.argmax(np.where(lev<lev_thr, p_mean, -1)))
i_bord = int(np.argmin(np.abs(p_mean-0.5)))
i_ood  = int(np.argmax(lev))
presets = []
for name,i in [('low risk',i_low),('borderline',i_bord),('high risk',i_high),('outside the data',i_ood)]:
    presets.append({'name':name,'x':[round(float(v),1) for v in X[i]],'y':int(y[i])})
    print(name, X[i], 'p=%.2f'%p_mean[i], 'lev=%.1f'%lev[i])

out = {
 'feats':FEATS,'labels':[LABELS[f] for f in FEATS],'actionable':[f in ACTIONABLE for f in FEATS],
 'mu':[round(float(v),3) for v in mu],'sd':[round(float(v),3) for v in sd],
 'ranges':[[round(float(X[:,j].min()),1), round(float(X[:,j].max()),1)] for j in range(k)],
 'W':[[round(float(v),4) for v in wrow] for wrow in W],
 'w0':[round(float(v),4) for v in w0],
 'cal':cal,'auc':round(auc,3),'brier':round(brier,3),'n':int(n),'pos_rate':round(float(y.mean()),3),
 'lev_thr':round(lev_thr,2),'unc_thr':round(unc_thr,4),'sweep':sweep,'presets':presets,
}
json.dump(out, open('model.json','w'))
print('model.json written,', len(json.dumps(out)), 'bytes')
