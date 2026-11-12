const byId=id=>document.getElementById(id);
const chicago=new Intl.DateTimeFormat('en-US',{timeZone:'America/Chicago',dateStyle:'medium',timeStyle:'long'});
async function refresh(){
  try{
    const response=await fetch('/reports/batch-restoration',{cache:'no-store'});
    if(!response.ok)throw Error(`Observation unavailable (${response.status})`);
    const value=await response.json();const observed=new Date(value.observed_at);
    const stale=Date.now()-observed.getTime()>180000;
    byId('observation').textContent=`Observed ${chicago.format(observed)}${stale?' — STALE':''}`;
    byId('scope').textContent=value.scope;
    byId('checks').replaceChildren(...value.checks.map(check=>{const p=document.createElement('p');p.textContent=`${check.name}: ${check.status}. ${check.detail}`;if(check.status!=='verified')p.className='pending';return p;}));
    for(const episode of value.episodes){
      let card=byId(episode.id);
      if(!card){card=document.createElement('article');card.id=episode.id;const h=document.createElement('h2');const p=document.createElement('p');const video=document.createElement('video');video.controls=true;video.preload='metadata';video.playsInline=true;video.src=episode.url;const link=document.createElement('a');link.href=episode.url;link.textContent='Open video';card.append(h,p,video,document.createElement('br'),link);byId('episodes').append(card);}
      card.querySelector('h2').textContent=episode.title;card.querySelector('p').textContent=episode.description;
    }
    byId('evidence').textContent=JSON.stringify(value,null,2);
  }catch(error){byId('observation').textContent=`UNAVAILABLE: ${error.message}; previous observations may be stale.`;}
}
refresh();setInterval(refresh,30000);
