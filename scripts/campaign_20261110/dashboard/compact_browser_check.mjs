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
const first=await evaluate(`({refreshes:+document.body.dataset.refreshCount,origin:performance.timeOrigin,nav:[...document.querySelectorAll('nav a')].map(x=>x.textContent),visible:[...document.querySelectorAll('main > section.selected-view')].map(x=>x.id),error:document.querySelector('#fetch-error').innerText,width:document.documentElement.scrollWidth,viewport:innerWidth,counts:counts(current),text:document.querySelector('#overview').innerText})`);
assert.deepEqual(first.nav,['Overview','Compute','Active Work']);assert.deepEqual(first.visible,['overview']);assert.equal(first.error,'');assert.ok(first.width<=first.viewport);assert.ok(first.text.includes('Scope incomplete'));assert.ok(first.text.includes('not frozen completion'));
for(const view of ['overview','compute','active-work','runtime-validation']){
 await evaluate(`location.hash=${JSON.stringify(view)}`);await delay(150);
 assert.deepEqual(await evaluate(`[...document.querySelectorAll('main > section.selected-view')].map(x=>x.id)`),[view]);
 if(view!=='runtime-validation'){const shot=await call('Page.captureScreenshot',{format:'png'});await writeFile(output+'/'+view+'.png',Buffer.from(shot.data,'base64'));}
}
await evaluate(`location.hash='active-work';document.querySelector('#work-account').value='minyeong';document.querySelector('#work-account').dispatchEvent(new Event('change'));document.querySelector('#work-search').value='axial';document.querySelector('#work-search').dispatchEvent(new Event('input'));`);
await delay(31500);
const refreshed=await evaluate(`({refreshes:+document.body.dataset.refreshCount,origin:performance.timeOrigin,account:document.querySelector('#work-account').value,search:document.querySelector('#work-search').value,view:location.hash})`);
assert.ok(refreshed.refreshes>first.refreshes);assert.equal(refreshed.origin,first.origin);assert.equal(refreshed.account,'minyeong');assert.equal(refreshed.search,'axial');assert.equal(refreshed.view,'#active-work');
await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
for(const view of ['overview','compute','active-work']){await evaluate(`location.hash=${JSON.stringify(view)}`);await delay(100);assert.ok(await evaluate('document.documentElement.scrollWidth<=innerWidth'),'Mobile overflow: '+view);}
const mobile=await call('Page.captureScreenshot',{format:'png'});await writeFile(output+'/mobile.png',Buffer.from(mobile.data,'base64'));
const states=await evaluate(`({stale:evidenceState('Active',Date.now()/1000-1000),missing:evidenceState('Active',null),workers:current.interactive_activity.workers.length,counts:counts(current),displayed:document.querySelector('#compact-counts').innerText})`);assert.equal(states.stale,'Stale');assert.equal(states.missing,'Unavailable');assert.equal(states.workers,4);
assert.equal(errors.length,0,JSON.stringify(errors));assert.ok(requests.filter(r=>r.url.startsWith('http://127.0.0.1:8765')).every(r=>r.method==='GET'));
await writeFile(output+'/browser_check.json',JSON.stringify({passed:true,observed_at:new Date().toISOString(),first,refreshed,states,requests:requests.length,errors},null,2));console.log(JSON.stringify({passed:true,refreshes:refreshed.refreshes,errors:errors.length,workers:states.workers}));ws.close();
