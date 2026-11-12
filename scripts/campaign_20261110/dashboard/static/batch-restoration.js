const byId=id=>document.getElementById(id);
const chicago=new Intl.DateTimeFormat('en-US',{timeZone:'America/Chicago',dateStyle:'medium',timeStyle:'long'});
async function refreshEvidence(){
  try{
    const response=await fetch('/reports/batch-restoration',{cache:'no-store'});
    if(!response.ok)throw Error(`Observation unavailable (${response.status})`);
    const value=await response.json();const observed=new Date(value.observed_at);
    const stale=Date.now()-observed.getTime()>180000;
    byId('observation').textContent=`Observed ${chicago.format(observed)}${stale?' — STALE':''}`;
    byId('scope').textContent=value.scope;
    byId('checks').replaceChildren(...value.checks.map(check=>{const p=document.createElement('p');p.textContent=`${check.name}: ${check.status.replaceAll('_',' ')}. ${check.detail}`;if(check.status!=='verified')p.className='pending';return p;}));
    const matrix=value.matrix;
    byId('matrix-observation').textContent=matrix?`Control observations: ${observedText(matrix.observed_at)}. Submitted, completed, and verified remain separate.`:'UNAVAILABLE: no authoritative control observation; no restoration result inferred.';
    byId('matrix-controls').replaceChildren(...(matrix?.controls||[]).map(row=>{
      const p=document.createElement('p');
      p.textContent=`${row.stage||row.kind||'stage unavailable'} / ${row.control}: ${row.status||'status unavailable'}; job ${row.job_id||'unavailable'}; scheduler ${row.scheduler_state||'unavailable'}; verification ${row.verification_status||'not verified'}. ${row.next_action||''} Observed ${observedText(row.observed_at||matrix.observed_at)}.`;
      if(row.verification_status!=='verified')p.className='pending';return p;
    }));
    for(const episode of value.episodes){
      let card=byId(episode.id);
      if(!card){card=document.createElement('article');card.id=episode.id;const h=document.createElement('h2');const p=document.createElement('p');const video=document.createElement('video');video.controls=true;video.preload='metadata';video.playsInline=true;video.src=episode.url;const link=document.createElement('a');link.href=episode.url;link.textContent='Open video';card.append(h,p,video,document.createElement('br'),link);byId('episodes').append(card);}
      card.querySelector('h2').textContent=episode.title;card.querySelector('p').textContent=episode.description;
    }
    byId('evidence').textContent=JSON.stringify(value,null,2);
  }catch(error){byId('observation').textContent=`UNAVAILABLE: ${error.message}; previous observations may be stale.`;}
}
function observedText(value){
  const date=new Date(value);
  if(!value||!Number.isFinite(date.getTime()))return 'timestamp unavailable';
  return `${chicago.format(date)}${Date.now()-date.getTime()>180000?' — STALE':''}`;
}
async function refreshPipeline(){
  try{
    const response=await fetch('/reports/pipeline-status',{cache:'no-store'});
    if(!response.ok)throw Error(`HTTP ${response.status}`);
    const value=await response.json();
    if(value.campaign_id!=='campaign_20261110')throw Error('Campaign identity mismatch');
    const rows=(value.records||[]).filter(row=>row.batch?.startsWith('tray_changed_augmentation'));
    if(!rows.length)throw Error('Changed-augmentation records unavailable');
    const nodes=rows.map(row=>{
      const p=document.createElement('p');const review=row.review||{};
      const name=row.start===50303100?'Original A':row.start===50303200?'Original B':
        row.batch.includes('_a_')?'A repair':row.batch.includes('_b_')?'B repair':row.candidate_id;
      const status=row.status||'unavailable';
      const native=row.native?`native ${row.native.written} written, observed ${observedText(row.native.observed_at)}`:'native result unavailable';
      const demo=review.demo_completed!=null?`; demo ${review.demo_successful}/${review.demo_completed} successful`:'';
      const job=review.allocation?.jobid?`; collector ${review.allocation.jobid}`:'';
      const failureMessage=review.error||review.failure_evidence?.goal_occlusion?.first?.message||review.failure_evidence?.reasons?.join(', ');
      const failure=failureMessage?` Failed check: ${failureMessage}.`:'';
      const next=review.next_action?` Next action: ${review.next_action}.`:'';
      p.textContent=`${name} (${row.start}): ${status.replaceAll('_',' ')}. ${native}${demo}${job}.${failure}${next} Stage observation: ${observedText(review.observed_at||row.last_observed_at)}. Owner: dataset_contract; dispatch: compute.`;
      p.className='pending';return p;
    });
    byId('live-pipeline').replaceChildren(...nodes);
  }catch(error){byId('live-pipeline').textContent=`UNAVAILABLE: ${error.message}; no restoration result inferred.`;}
}
async function refresh(){await Promise.allSettled([refreshEvidence(),refreshPipeline()]);}
refresh();setInterval(refresh,30000);
