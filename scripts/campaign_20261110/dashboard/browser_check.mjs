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
const first=await evaluate(`({refreshes:+document.body.dataset.refreshCount,origin:performance.timeOrigin,counts:current.evidence.counts,rows:document.querySelectorAll('#coverage-table tbody tr').length,totals:document.querySelector('#totals').innerText,error:document.querySelector('#fetch-error').innerText,viewport:innerWidth,width:document.documentElement.scrollWidth})`);
assert.ok(first.refreshes>=1,'First data fetch rendered');assert.ok(first.rows>=322,'Complete coverage denominator rendered');assert.equal(first.error,'');for(const [split,value] of Object.entries(first.counts)){assert.equal(await evaluate(`document.querySelector('[data-split="${split}"]').childNodes[0].textContent.trim()`),new Intl.NumberFormat('en-US').format(value));}assert.ok(first.width<=first.viewport,'Desktop has no page overflow');
await evaluate(`document.querySelector('#split').value='eval_model';document.querySelector('#split').dispatchEvent(new Event('change'));document.querySelector('#search').value='disembedding';document.querySelector('#search').dispatchEvent(new Event('input'))`);
const filtered=await evaluate(`({rows:document.querySelectorAll('#coverage-table tbody tr').length,text:document.querySelector('#coverage-table tbody').innerText})`);
assert.ok(filtered.rows>0&&filtered.rows<first.rows);assert.ok(filtered.text.includes('Model holdout'));
console.log(JSON.stringify({phase:'initial-render',counts:first.counts,coverageRows:first.rows,filteredRows:filtered.rows}));
await delay(31500);
const refreshed=await evaluate(`({refreshes:+document.body.dataset.refreshCount,origin:performance.timeOrigin,filter:document.querySelector('#search').value,split:document.querySelector('#split').value})`);
assert.ok(refreshed.refreshes>first.refreshes,'30-second automatic refresh occurred');assert.equal(refreshed.origin,first.origin,'Page did not reload');assert.equal(refreshed.filter,'disembedding');assert.equal(refreshed.split,'eval_model');
await evaluate(`document.querySelector('#split').value='all';document.querySelector('#search').value='';render(current)`);
const screenshot=await call('Page.captureScreenshot',{format:'png'});await writeFile(output+'/desktop.png',Buffer.from(screenshot.data,'base64'));
const states=await evaluate(`(()=>{const stale=observation('Test',Date.now()/1000-1000,true,null);const missing=observation('Test',null,false,null);const bad=observation('Test',Date.now()/1000,true,'probe failed');return {stale:stale.includes('Stale'),missing:missing.includes('Unavailable'),bad:bad.includes('Failed'),chicago:date('2026-11-12T19:00:00Z')};})()`);
assert.ok(states.stale&&states.missing&&states.bad);assert.ok(states.chicago.includes('1:00:00 PM CST'));
await call('Network.setBlockedURLs',{urls:['*127.0.0.1:8765/api/status*']});
await evaluate('refresh()');assert.ok(await evaluate(`!document.querySelector('#fetch-error').classList.contains('hidden')`));
await call('Network.setBlockedURLs',{urls:[]});await evaluate('refresh()');assert.ok(await evaluate(`document.querySelector('#fetch-error').classList.contains('hidden')`));
await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});await delay(300);
const mobile=await evaluate('({viewport:innerWidth,width:document.documentElement.scrollWidth})');assert.ok(mobile.width<=mobile.viewport,'Mobile has no page overflow');
const mobileShot=await call('Page.captureScreenshot',{format:'png'});await writeFile(output+'/mobile.png',Buffer.from(mobileShot.data,'base64'));
await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});await evaluate(`document.querySelector('#compute').scrollIntoView({behavior:'instant'})`);await delay(200);const computeShot=await call('Page.captureScreenshot',{format:'png'});await writeFile(output+'/compute.png',Buffer.from(computeShot.data,'base64'));await evaluate(`document.querySelector('#activity').scrollIntoView({behavior:'instant'})`);await delay(200);const activityShot=await call('Page.captureScreenshot',{format:'png'});await writeFile(output+'/activity.png',Buffer.from(activityShot.data,'base64'));await evaluate('scrollTo(0,0)');
assert.equal(errors.length,0,'No JavaScript runtime exceptions');assert.ok(requests.filter(r=>r.url.startsWith('http://127.0.0.1:8765')).every(r=>r.method==='GET'),'Browser sent GET only');
const result={passed:true,observed_at:new Date().toISOString(),first,refreshed,filtered,states,mobile,request_count:requests.length,errors};
await writeFile(output+'/browser_check.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));ws.close();
