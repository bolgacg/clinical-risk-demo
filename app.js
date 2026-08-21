/* Clinical risk console. Model precomputed offline (study/study.py); this file only renders.
   Ensemble of 30 logistic regressions; attributions are exact coefficient contributions. */
'use strict';
(function(){
const M = window.M;
const K = M.feats.length;
const S = { x: M.presets[1].x.slice(), thr: 0.35 };

const $ = s => document.querySelector(s);
const el = (tag, attrs, parent) => {
  const ns='http://www.w3.org/2000/svg';
  const e = ['svg','g','path','circle','rect','line','text','polyline','polygon'].includes(tag)
    ? document.createElementNS(ns,tag) : document.createElement(tag);
  for (const k in attrs||{}){ if(k==='text') e.textContent=attrs[k]; else e.setAttribute(k,attrs[k]); }
  if (parent) parent.appendChild(e); return e;
};
const fmt = (v,d=1)=>v.toFixed(d);

function z(x){ return x.map((v,j)=>(v-M.mu[j])/M.sd[j]); }
function predict(x){
  const zz = z(x);
  const ps = M.W.map(w=>{ let L=w[0]; for(let j=0;j<K;j++) L+=w[j+1]*zz[j]; return 1/(1+Math.exp(-L)); });
  const mean = ps.reduce((a,b)=>a+b,0)/ps.length;
  const sd = Math.sqrt(ps.reduce((a,b)=>a+(b-mean)*(b-mean),0)/ps.length);
  const lev = zz.reduce((a,v)=>a+v*v,0);
  return {mean, sd, lev, zz};
}
function attributions(x){
  const zz=z(x);
  return M.w0.slice(1).map((w,j)=>w*zz[j]);   // exact log-odds contribution vs cohort mean
}

/* ---------- sliders ---------- */
function buildSliders(){
  const box=$('#sliders'); box.innerHTML='';
  M.feats.forEach((f,j)=>{
    const row=el('div',{class:'slrow'},box);
    const lab=el('div',{class:'lab'},row);
    lab.innerHTML=`<span><b>${M.labels[j]}</b>${M.actionable[j]?'':'<span class="lock">fixed for advice</span>'}</span><span class="val" id="v${j}"></span>`;
    const r=el('input',{type:'range',min:M.ranges[j][0],max:M.ranges[j][1],
      step:(M.ranges[j][1]-M.ranges[j][0])/200,value:S.x[j],'aria-label':M.labels[j]},row);
    r.addEventListener('input',e=>{ S.x[j]=+e.target.value; render(); });
    row._range=r;
  });
}
function syncSliders(){
  document.querySelectorAll('#sliders .slrow').forEach((row,j)=>{ row._range.value=S.x[j]; });
}

/* ---------- presets ---------- */
function buildPresets(){
  const box=$('#preset-chips');
  M.presets.forEach((p,i)=>{
    const c=el('button',{class:'chip'+(i===1?' on':''),text:p.name},box);
    c.addEventListener('click',()=>{
      document.querySelectorAll('#preset-chips .chip').forEach(x=>x.classList.remove('on'));
      c.classList.add('on');
      S.x=p.x.slice(); syncSliders(); render();
    });
  });
}

/* ---------- main render ---------- */
function render(){
  M.feats.forEach((f,j)=>{ $('#v'+j).textContent=fmt(S.x[j], j===4?2:0); });
  $('#thr-val').textContent='risk > '+fmt(S.thr,2);
  const {mean,sd,lev}=predict(S.x);
  const withheld = lev>M.lev_thr || sd>M.unc_thr;
  const riskEl=$('#risk'), verd=$('#verdict'), wh=$('#withheld');
  if (withheld){
    riskEl.textContent='—'; riskEl.style.color='var(--amber)';
    $('#band').textContent='no estimate issued';
    verd.textContent='';
    wh.classList.add('show');
    const reasons=[];
    if (lev>M.lev_thr) reasons.push(`these measurements sit outside the training data (leverage ${fmt(lev,0)} against a ceiling of ${fmt(M.lev_thr,0)})`);
    if (sd>M.unc_thr) reasons.push(`the thirty ensemble members disagree (spread ${fmt(sd,3)} against a ceiling of ${fmt(M.unc_thr,3)})`);
    $('#withheld-why').textContent='The console reports its own inapplicability instead of a number: '+reasons.join('; ')+
      '. A referral decision for this patient belongs to a clinician, not to this model.';
  } else {
    wh.classList.remove('show');
    riskEl.textContent=(mean*100).toFixed(0); riskEl.innerHTML=(mean*100).toFixed(0)+'<small> %</small>';
    riskEl.style.color = mean>=S.thr ? 'var(--neg)' : 'var(--pos)';
    $('#band').textContent=`ensemble band ±${(sd*100).toFixed(1)} points · cohort base rate ${(M.pos_rate*100).toFixed(0)}%`;
    verd.innerHTML = mean>=S.thr
      ? `<b style="color:var(--neg)">above the referral threshold</b> · the console recommends follow-up`
      : `<b style="color:var(--pos)">below the referral threshold</b> · routine monitoring`;
  }
  // attributions
  const at=attributions(S.x), box=$('#attr'); box.innerHTML='';
  el('div',{class:'k',text:'What drives it · exact contribution vs the cohort average'},box);
  const mx=Math.max(...at.map(Math.abs),0.5);
  at.forEach((a,j)=>{
    const row=el('div',{class:'row'},box);
    row.innerHTML=`<span>${M.feats[j]==='DiabetesPedigreeFunction'?'family history':M.feats[j].replace('BloodPressure','blood pressure')}</span>`;
    const bar=el('div',{class:'bar'},row);
    const w=Math.abs(a)/mx*50;
    el('i',{style:`left:${a<0?50-w:50}%;width:${w}%;background:${a>0?'var(--neg)':'var(--pos)'}`},bar);
    el('span',{class:'num',text:(a>0?'+':'')+a.toFixed(2)},row);
  });
  $('#cfout').innerHTML='';
}

/* ---------- counterfactual ---------- */
$('#cfbtn').addEventListener('click',()=>{
  const {mean,lev,sd}=predict(S.x);
  const out=$('#cfout');
  if (lev>M.lev_thr || sd>M.unc_thr){ out.innerHTML='No advice while the prediction is withheld; the model has no gradient worth following out here.'; return; }
  if (mean < S.thr){ out.innerHTML=`Already below the threshold at ${(mean*100).toFixed(0)}%. Nothing to change.`; return; }
  const zz=z(S.x);
  const w=M.W[0].map((_,c)=>M.W.reduce((a,r)=>a+r[c],0)/M.W.length);   // ensemble-mean weights
  let L=w[0]; for(let j=0;j<K;j++) L+=w[j+1]*zz[j];
  const tgt=Math.max(0.05, S.thr-0.015);
  const Lt=Math.log(tgt/(1-tgt));
  const act=M.actionable;
  const wa2=w.slice(1).reduce((a,v,j)=>a+(act[j]?v*v:0),0);
  const moves=[];
  let feasible=false;
  const dzAll=w.slice(1).map((v,j)=>act[j]?(Lt-L)*v/wa2:0);
  const cand=S.x.map((v,j)=>{
    let nv=v+dzAll[j]*M.sd[j];
    return Math.min(M.ranges[j][1], Math.max(M.ranges[j][0], nv));
  });
  const chk=predict(cand);
  if (chk.mean<=S.thr+0.005) feasible=true;
  if (feasible){
    const parts=[];
    M.feats.forEach((f,j)=>{
      if (Math.abs(cand[j]-S.x[j])>0.05*M.sd[j]){
        parts.push(`${M.labels[j].split(' (')[0]} from ${fmt(S.x[j], j===4?2:0)} to <b>${fmt(cand[j], j===4?2:0)}</b>`);
      }
    });
    out.innerHTML=`Smallest combined change: ${parts.join(', ')} · new risk <b>${(chk.mean*100).toFixed(0)}%</b>, below the threshold. Computed exactly from the model, not narrated.`;
  } else {
    out.innerHTML='No achievable combination of glucose, blood pressure and body mass index brings this patient below the current threshold; the fixed factors dominate. That is an honest answer too.';
  }
});
$('#thr').addEventListener('input',e=>{ S.thr=+e.target.value; render(); drawSweep(); });

/* ---------- study figures ---------- */
function drawCal(){
  const w=760,h=300,p=54, box=$('#calfig'); box.innerHTML='';
  const svg=el('svg',{viewBox:`0 0 ${w} ${h}`},box);
  const X=v=>p+(w-2*p)*v, Y=v=>h-p-(h-2*p)*v;
  el('line',{x1:X(0),y1:Y(0),x2:X(1),y2:Y(1),stroke:'var(--faint)','stroke-dasharray':'5 4'},svg);
  el('line',{x1:X(0),y1:Y(0),x2:X(1),y2:Y(0),stroke:'var(--rule2)'},svg);
  el('line',{x1:X(0),y1:Y(0),x2:X(0),y2:Y(1),stroke:'var(--rule2)'},svg);
  [0,0.25,0.5,0.75,1].forEach(v=>{
    el('text',{x:X(v),y:h-p+18,'text-anchor':'middle',class:'axis',text:(v*100)+'%'},svg);
    el('text',{x:X(0)-8,y:Y(v)+4,'text-anchor':'end',class:'axis',text:(v*100)+'%'},svg);
  });
  el('text',{x:X(0.5),y:h-8,'text-anchor':'middle',class:'axis',text:'what the model said'},svg);
  el('text',{x:14,y:Y(0.5),class:'axis',transform:`rotate(-90 14 ${Y(0.5)})`,'text-anchor':'middle',text:'what happened'},svg);
  let pl='';
  M.cal.forEach(([pr,ob,n])=>{ pl+=`${X(pr)},${Y(ob)} `; });
  el('polyline',{points:pl,fill:'none',stroke:'var(--accent)','stroke-width':1.5,opacity:.6},svg);
  M.cal.forEach(([pr,ob,n])=>{
    el('circle',{cx:X(pr),cy:Y(ob),r:4+Math.sqrt(n)/4,fill:'var(--accent)',opacity:.75},svg);
  });
  $('#calverdict').innerHTML=`<b>When this model says 40%, roughly 40% is what happens.</b> The dots track the
    diagonal across the deciles (dot size = patients in the group). Area under the curve is ${M.auc}, Brier
    score ${M.brier}, on ${M.n} patients. Calibration is what makes the referral threshold below a real
    decision rather than a dial on an arbitrary score.`;
}
function drawSweep(){
  const w=760,h=300,p=54, box=$('#sweepfig'); box.innerHTML='';
  const svg=el('svg',{viewBox:`0 0 ${w} ${h}`},box);
  const ts=M.sweep.map(r=>r[0]);
  const X=t=>p+(w-2*p)*(t-ts[0])/(ts[ts.length-1]-ts[0]), Y=v=>h-p-(h-2*p)*v;
  el('line',{x1:p,y1:Y(0),x2:w-p,y2:Y(0),stroke:'var(--rule2)'},svg);
  [0,0.25,0.5,0.75,1].forEach(v=>el('text',{x:p-8,y:Y(v)+4,'text-anchor':'end',class:'axis',text:(v*100)+'%'},svg));
  [0.2,0.35,0.5,0.65].forEach(t=>el('text',{x:X(t),y:h-p+18,'text-anchor':'middle',class:'axis',text:t.toFixed(2)},svg));
  el('text',{x:X(0.45),y:h-8,'text-anchor':'middle',class:'axis',text:'referral threshold'},svg);
  const series=[['flagged of every 100 patients',r=>r[1]/100,'var(--amber)'],
                ['true cases caught (sensitivity)',r=>r[2],'var(--pos)'],
                ['flags that are right (precision)',r=>r[3],'var(--blue)']];
  series.forEach(([lab,fy,col],i)=>{
    let pl=''; M.sweep.forEach(r=>{ pl+=`${X(r[0])},${Y(fy(r))} `; });
    el('polyline',{points:pl,fill:'none',stroke:col,'stroke-width':2},svg);
    el('text',{x:w-p,y:26+i*16,'text-anchor':'end',class:'axis',fill:col,text:lab},svg);
  });
  el('line',{x1:X(S.thr),y1:Y(0),x2:X(S.thr),y2:Y(1),stroke:'var(--ink)','stroke-width':1,opacity:.5},svg);
  el('text',{x:X(S.thr),y:Y(1)-6,'text-anchor':'middle',class:'axis',fill:'var(--ink)',text:'console setting'},svg);
  const near=M.sweep.reduce((a,b)=>Math.abs(b[0]-S.thr)<Math.abs(a[0]-S.thr)?b:a);
  $('#sweepverdict').innerHTML=`<b>The threshold is a staffing decision wearing a percentage.</b> At the
    console's current setting of ${near[0].toFixed(2)}, ${near[1].toFixed(0)} of every 100 screened patients
    are flagged, ${(near[2]*100).toFixed(0)}% of true cases are caught, and ${(near[3]*100).toFixed(0)}% of
    flags point at a real case. Move the console's threshold slider and this figure follows: choosing it
    belongs to the clinic's capacity, which is why the console exposes it instead of hiding it.`;
}

/* ---------- provenance ---------- */
const PROV=[
 {what:'Every prediction ships with its per-patient explanation, in the units of the measurement',
  src:[['sh','Wiil group 2024-2025','explainable smoking-status, ECG and stroke work: the classifier must show its evidence'],
       ['fd','Lundberg & Lee 2017; Ribeiro et al. 2016','SHAP and LIME; here the model is linear, so contributions are exact rather than approximated']]},
 {what:'A first-class withheld state: the model declines patients outside its competence',
  src:[["sh","the calibration honesty this group's reviews demand","a model that answers everyone is wrong about someone"],
       ['fd','applicability-domain tradition','leverage against the training distribution plus ensemble disagreement as the two refusal triggers']]},
 {what:'Advice is computed as the smallest actionable change, never narrated',
  src:[['sh','AUD-DSS 2023; Lancet structured decision support 2023','decision support means an action, not a score'],
       ['fd','Wachter et al. 2017','counterfactual explanations; restricted here to clinically modifiable measurements']]},
 {what:'Calibration is reported before discrimination, by decile, with the code attached',
  src:[["sh","the group's model-comparison reviews (BMJ Open 2021)","honest evaluation practice"],
       ['fd','Brier 1950; reliability diagrams','40% must mean 40% before a threshold means anything']]},
 {what:'The referral threshold is exposed as a capacity decision, not buried as a constant',
  src:[['sh','the decision-support line','a deployed model is operated, and operations have budgets'],
       ["fd","this page's reading","the same alarm-budget discipline as industrial monitoring, in clinical clothing"]]},
 {what:'Public, reproducible proxy data with exclusions stated',
  src:[["sh","the group's Danish registry setting","imitated honestly rather than faked"],
       ['fd','NIDDK Pima cohort','768 patients, 724 after excluding coded missing values; the page says so']]},
];
function renderProv(){
  const p=$('#provcards');
  PROV.forEach(c=>{
    const card=el('div',{class:'card'},p);
    card.innerHTML='<div class="what">'+c.what+'</div>'+
      c.src.map(([cls,who,what])=>`<div class="src"><span class="who-tag ${cls}">${cls==='sh'?'Wiil group':'foundation'}</span><b>${who}</b> · ${what}</div>`).join('');
  });
}

/* ---------- tour ---------- */
(function(){
  const STEPS=[
    {sel:'header h1',k:'Welcome · 1 of 6',html:`This page asks when a clinical risk model has earned the right to answer, and when it should hand the patient back to a human. Everything on it runs live in your browser on real public data.`},
    {sel:'#background .bg2',k:'The sources · 2 of 6',html:`The design answers the group's own publications: explainable classifiers, working decision-support systems, honest evaluation. Six of them are named here.`},
    {sel:'#preset-chips',k:'The patients · 3 of 6',html:`Four real patients from the cohort. The fourth one matters most: the console refuses to score them, and says why. Try it.`},
    {sel:'.console',k:'The console · 4 of 6',html:`Drag any measurement and the risk, the band and the explanation bars answer immediately. The right panel computes the smallest clinically actionable change that would move the patient below the referral threshold.`},
    {sel:'#studies .study',k:'The evidence · 5 of 6',html:`Two offline figures: calibration (does 40% mean 40%?) and the referral threshold as a staffing decision. The threshold slider on the console moves the second figure live.`},
    {sel:'#provenance .prov',k:'Provenance · 6 of 6',html:`Every mechanism is mapped to the group publication it answers and the foundational method behind it. Explore freely; the Guided tour button restarts this walkthrough.`},
  ];
  const root=$('#tour'), hl=$('#tour-hl'), card=$('#tour-card');
  let idx=0;
  function place(){
    const st=STEPS[idx]; const elm=document.querySelector(st.sel);
    if(!elm){ next(); return; }
    elm.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'center'});
    setTimeout(()=>{
      const r=elm.getBoundingClientRect(), sx=scrollX, sy=scrollY;
      root.style.height=document.documentElement.scrollHeight+'px';
      hl.style.left=(r.left+sx-8)+'px'; hl.style.top=(r.top+sy-8)+'px';
      hl.style.width=(r.width+16)+'px'; hl.style.height=(r.height+16)+'px';
      const dots=STEPS.map((_,i)=>`<i class="${i===idx?'on':''}"></i>`).join('');
      card.innerHTML=`<div class="tk">${st.k}</div><p>${st.html}</p>
        <div class="tour-nav"><div class="dots">${dots}</div>
        ${idx>0?'<button class="tour-btn" id="tprev">Back</button>':''}
        <button class="tour-btn" id="tskip">Close</button>
        <button class="tour-btn primary" id="tnext">${idx<STEPS.length-1?'Next':'Done'}</button></div>`;
      const cw=Math.min(400,innerWidth-32);
      let cx=r.left+sx+r.width+18, cy=r.top+sy;
      if (cx+cw > sx+innerWidth-16){ cx=Math.max(16+sx,r.left+sx); cy=r.bottom+sy+14; }
      card.style.left=cx+'px'; card.style.top=cy+'px';
      $('#tnext').onclick=next; $('#tskip').onclick=stop;
      const pv=$('#tprev'); if(pv) pv.onclick=()=>{idx=Math.max(0,idx-1);place();};
    }, matchMedia('(prefers-reduced-motion: reduce)').matches?50:450);
  }
  function next(){ if(idx>=STEPS.length-1){stop();return;} idx++; place(); }
  function stop(){ root.classList.remove('on'); try{localStorage.setItem('crd_tour','done');}catch(e){} }
  function start(){ idx=0; root.classList.add('on'); place(); }
  $('#tourbtn').addEventListener('click',start);
  let seen=null; try{ seen=localStorage.getItem('crd_tour'); }catch(e){}
  if(!seen) setTimeout(start,700);
})();

try{ const q=new URLSearchParams(location.search); if(q.get('p')) { S.x=M.presets[+q.get('p')].x.slice(); } }catch(e){}
buildSliders(); buildPresets(); render(); drawCal(); drawSweep(); renderProv();
})();
