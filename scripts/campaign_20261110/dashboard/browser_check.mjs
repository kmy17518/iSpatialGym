// Manual QA against an isolated local Chrome debugging session. No job operations.
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';

const output=process.argv[2];
if(!output)throw new Error('Pass an output directory for screenshots and QA evidence');
const tabs=await (await fetch('http://127.0.0.1:9227/json/list')).json();
const ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);
await new Promise(resolve=>ws.addEventListener('open',resolve,{once:true}));
let next=0;
const pending=new Map(),errors=[],requests=[];
ws.addEventListener('message',event=>{
  const message=JSON.parse(event.data);
  if(message.id){const p=pending.get(message.id);pending.delete(message.id);message.error?p.reject(message.error):p.resolve(message.result);}
  if(message.method==='Runtime.exceptionThrown')errors.push(message.params.exceptionDetails);
  if(message.method==='Network.requestWillBeSent')requests.push(message.params.request);
});
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
await call('Runtime.enable');await call('Network.enable');await call('Page.enable');
await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
await call('Page.navigate',{url:'http://127.0.0.1:8765/'});
for(let i=0;i<40;i++){await delay(250);if(await evaluate('document.body?.dataset.refreshCount'))break;}
const first=await evaluate(`({refreshes:+document.body.dataset.refreshCount,origin:performance.timeOrigin,counts:counts(current),historicalCounts:rawCounts(current),rows:document.querySelectorAll('#coverage-table tbody tr').length,totals:document.querySelector('#totals').innerText,error:document.querySelector('#fetch-error').innerText,viewport:innerWidth,width:document.documentElement.scrollWidth})`);
assert.ok(first.refreshes>=1,'First data fetch rendered');assert.ok(first.rows>=322,'Complete coverage denominator rendered');assert.equal(first.error,'');for(const [split,value] of Object.entries(first.counts)){assert.equal(await evaluate(`document.querySelector('[data-split="${split}"]').childNodes[0].textContent.trim()`),value===null?'—':new Intl.NumberFormat('en-US').format(value));}assert.ok(first.width<=first.viewport,'Desktop has no page overflow');
const training=await evaluate(`({gate:current.training_production_gate,eligibility:current.training_eligibility,missing:counts({...current,training_eligibility:{}}).train,onlyEligible:counts({...current,training_eligibility:{eligible_accepted:7}}).train,historyRates:rateRows({...current,evidence:{accepted_history:[{observed_at:'2026-11-11T00:00:00Z',counts:{train:1000}},{observed_at:'2026-11-11T01:00:00Z',counts:{train:5000}}]}}).rates.train,text:document.querySelector('#training-production').innerText})`);
assert.equal(training.missing,null,'Missing eligibility never falls back to historical counts');
assert.equal(training.onlyEligible,7,'Only eligible training reaches goal progress');
assert.equal(training.historyRates,null,'Raw historical training growth never becomes eligible throughput');
assert.ok(training.text.includes('Flagged / excluded')&&training.text.includes('Compliant replacements'));
const recovery=await evaluate(`({rows:document.querySelectorAll('#recovery-rows tr').length,records:current.standard_recovery.classes.length,summary:current.standard_recovery.summary,text:document.querySelector('#recovery-summary').innerText})`);
assert.equal(recovery.rows,recovery.records,'Every recovery class is rendered');
assert.ok(recovery.rows>=59,'The complete recovery baseline remains visible');
await evaluate(`document.querySelector('#recovery-search').value='disembedding';document.querySelector('#recovery-search').dispatchEvent(new Event('input'));document.querySelector('#recovery-owner').value='/root';document.querySelector('#recovery-owner').dispatchEvent(new Event('change'))`);
assert.ok(await evaluate(`document.querySelectorAll('#recovery-rows tr').length>0&&document.querySelector('#recovery-rows').innerText.includes('disembedding')`),'Recovery filters find the assigned classes');
const goalYield=await evaluate(`({observed_at:current.group_yield.observed_at,summary:current.group_yield.summary,reconciliation:current.group_yield.reconciliation,errors:current.group_yield.errors,rows:document.querySelectorAll('#yield-rows tr').length,text:document.querySelector('#yield-summary').innerText})`);
assert.ok(goalYield.rows>10,'Goal-group cohorts rendered');assert.equal(goalYield.errors.length,0,'Authoritative group audit has no errors');assert.equal(goalYield.reconciliation.group_accepted_instances,goalYield.reconciliation.noncamera_committed_instances,'Group members reconcile with SQL');
for(const [mode,row] of Object.entries(goalYield.summary)){assert.equal(await evaluate(`document.querySelector('[data-yield-completed="${mode}"]').childNodes[0].textContent.trim()`),new Intl.NumberFormat('en-US').format(row.completed_groups));assert.equal(Object.values(row.success_histogram).reduce((a,b)=>a+b,0),row.completed_groups);}
assert.ok(goalYield.text.includes('Exact historical attempt denominator unavailable'));
assert.ok(goalYield.text.includes('native draws'),'Retry-inclusive attempt denominator is explicit');
const fleetTime=await evaluate(`({available:current.sources.fleet_gpu_time.available,text:document.querySelector('#fleet-gpu-time').innerText,seconds:current.fleet_gpu_time.fleet_window_estimated_gpu_seconds})`);
assert.ok(fleetTime.available&&fleetTime.seconds>=0,'Authoritative fleet GPU time is available');
assert.ok(fleetTime.text.includes('not divided by this partial time window'),'No mixed-window yield inference');
const recordedFleet=await evaluate('current.fleet_gpu_time.recorded_fleet_estimated_gpu_seconds');
assert.ok(recordedFleet>=fleetTime.seconds,'Recovered rental history contains the instrumented window');
assert.ok(fleetTime.text.includes('All recorded campaign rentals')&&fleetTime.text.includes((recordedFleet/3600).toFixed(2)),'Recovered rental hours and scope rendered');
const clusterTime=await evaluate(`({available:current.sources.cluster_fleet_gpu_time.available,text:document.querySelector('#cluster-fleet-gpu-time').innerText,seconds:current.cluster_fleet_gpu_time.verified_allocated_gpu_seconds})`);
assert.ok(clusterTime.available&&clusterTime.seconds>=0,'Verified cluster allocation accounting is available');
assert.ok(clusterTime.text.includes('no accepted count or throughput is inferred'),'Broader allocation time remains separate from group yield');
await evaluate(`document.querySelector('#yield-search').value='mental_rotation';document.querySelector('#yield-search').dispatchEvent(new Event('input'));document.querySelector('#yield-mode').value='train_only';document.querySelector('#yield-mode').dispatchEvent(new Event('change'))`);
assert.ok(await evaluate(`document.querySelectorAll('#yield-rows tr').length>0&&document.querySelector('#yield-rows').innerText.includes('mental_rotation')`));
await evaluate(`document.querySelector('#split').value='eval_model';document.querySelector('#split').dispatchEvent(new Event('change'));document.querySelector('#search').value='disembedding';document.querySelector('#search').dispatchEvent(new Event('input'))`);
const filtered=await evaluate(`({rows:document.querySelectorAll('#coverage-table tbody tr').length,text:document.querySelector('#coverage-table tbody').innerText})`);
assert.ok(filtered.rows>0&&filtered.rows<first.rows);assert.ok(filtered.text.includes('Model holdout'));
console.log(JSON.stringify({phase:'initial-render',counts:first.counts,coverageRows:first.rows,filteredRows:filtered.rows}));
await delay(31500);
const refreshed=await evaluate(`({refreshes:+document.body.dataset.refreshCount,origin:performance.timeOrigin,filter:document.querySelector('#search').value,split:document.querySelector('#split').value})`);
assert.ok(refreshed.refreshes>first.refreshes,'30-second automatic refresh occurred');assert.equal(refreshed.origin,first.origin,'Page did not reload');assert.equal(refreshed.filter,'disembedding');assert.equal(refreshed.split,'eval_model');
assert.equal(await evaluate(`document.querySelector('#yield-search').value`),'mental_rotation');assert.equal(await evaluate(`document.querySelector('#yield-mode').value`),'train_only');
assert.equal(await evaluate(`document.querySelector('#recovery-search').value`),'disembedding');assert.equal(await evaluate(`document.querySelector('#recovery-owner').value`),'/root');
await evaluate(`document.querySelector('#recovery-search').value='';document.querySelector('#recovery-owner').value='all';renderRecovery(current);document.querySelector('#standard-recovery').scrollIntoView({behavior:'instant'})`);await delay(200);const recoveryShot=await call('Page.captureScreenshot',{format:'png'});await writeFile(output+'/standard-recovery.png',Buffer.from(recoveryShot.data,'base64'));
await evaluate(`document.querySelector('#yield-search').value='';document.querySelector('#yield-mode').value='all';renderYield(current);document.querySelector('#goal-yield').scrollIntoView({behavior:'instant'})`);await delay(200);const yieldShot=await call('Page.captureScreenshot',{format:'png'});await writeFile(output+'/goal-yield.png',Buffer.from(yieldShot.data,'base64'));await evaluate('scrollTo(0,0)');
await evaluate(`document.querySelector('#split').value='all';document.querySelector('#search').value='';render(current)`);
await evaluate(`document.querySelector('#training-production').scrollIntoView({behavior:'instant'})`);await delay(200);const trainingShot=await call('Page.captureScreenshot',{format:'png'});await writeFile(output+'/training-production.png',Buffer.from(trainingShot.data,'base64'));await evaluate('scrollTo(0,0)');
const screenshot=await call('Page.captureScreenshot',{format:'png'});await writeFile(output+'/desktop.png',Buffer.from(screenshot.data,'base64'));
const states=await evaluate(`(()=>{const stale=observation('Test',Date.now()/1000-1000,true,null);const missing=observation('Test',null,false,null);const bad=observation('Test',Date.now()/1000,true,'probe failed');return {stale:stale.includes('Stale'),missing:missing.includes('Unavailable'),bad:bad.includes('Failed'),chicago:date('2026-11-12T19:00:00Z')};})()`);
assert.ok(states.stale&&states.missing&&states.bad);assert.ok(states.chicago.includes('1:00:00 PM CST'));
await call('Network.setBlockedURLs',{urls:['*127.0.0.1:8765/api/status*']});
await evaluate('refresh()');assert.ok(await evaluate(`!document.querySelector('#fetch-error').classList.contains('hidden')`));
await call('Network.setBlockedURLs',{urls:[]});await evaluate('refresh()');assert.ok(await evaluate(`document.querySelector('#fetch-error').classList.contains('hidden')`));
await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});await delay(300);
const mobile=await evaluate('({viewport:innerWidth,width:document.documentElement.scrollWidth})');assert.ok(mobile.width<=mobile.viewport,'Mobile has no page overflow');assert.equal(mobile.viewport,390,'Mobile layout retains the configured viewport width');
const mobileShot=await call('Page.captureScreenshot',{format:'png'});await writeFile(output+'/mobile.png',Buffer.from(mobileShot.data,'base64'));
await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});await evaluate(`document.querySelector('#compute').scrollIntoView({behavior:'instant'})`);await delay(200);const computeShot=await call('Page.captureScreenshot',{format:'png'});await writeFile(output+'/compute.png',Buffer.from(computeShot.data,'base64'));await evaluate(`document.querySelector('#activity').scrollIntoView({behavior:'instant'})`);await delay(200);const activityShot=await call('Page.captureScreenshot',{format:'png'});await writeFile(output+'/activity.png',Buffer.from(activityShot.data,'base64'));await evaluate('scrollTo(0,0)');
assert.equal(errors.length,0,'No JavaScript runtime exceptions');assert.ok(requests.filter(r=>r.url.startsWith('http://127.0.0.1:8765')).every(r=>r.method==='GET'),'Browser sent GET only');
const result={passed:true,observed_at:new Date().toISOString(),first,recovery,goalYield,training,fleetTime,clusterTime,refreshed,filtered,states,mobile,request_count:requests.length,errors};
await writeFile(output+'/browser_check.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));ws.close();
