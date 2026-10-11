const $ = id => document.getElementById(id);
let current = null, busy = false, playing = false, timer = null;
let decisionToken = null;
const decisionLabels = {acknowledge:'Đã xem', inspect:'Đề nghị kiểm tra', dismiss:'Bỏ qua'};
const nf = new Intl.NumberFormat('vi-VN', {maximumFractionDigits: 2, minimumFractionDigits: 2});
const number = value => value == null ? '—' : nf.format(value);
const timestamp = value => value ? `${value.slice(8,10)}/${value.slice(5,7)} ${value.slice(11,16)}` : '—';
const statuses = {forecast_ready:'Dữ liệu hợp lệ',invalid_history:'Lỗi lịch sử đầu vào',model_unavailable:'Không có model',prediction_failed:'Không thể dự báo',forecast_conflict:'Dự báo chưa thống nhất'};
function controls(){
  $('reset').disabled = busy; $('start').disabled = busy;
  $('step').disabled = busy || playing || !current?.session || current.finished;
  $('play').disabled = (!playing && busy) || !current?.session || current.finished;
  $('play').textContent = playing ? 'Ⅱ Dừng' : '▶ Chạy';
  $('playState').textContent = current?.finished ? 'Đã đi hết dữ liệu phiên.' : (playing ? 'Đang phát lại từng bước 15 phút' : 'Đang dừng');
  const canDecide = !busy && !playing && !!current?.current_forecast;
  for(const id of ['operator','decisionNote','inspect','dismiss']) $(id).disabled = !canDecide;
  $('acknowledge').disabled = !canDecide || current.current_forecast.status !== 'forecast_ready';
}
function stop(){playing=false; clearTimeout(timer); controls();}
function node(name,attrs,text){const el=document.createElementNS('http://www.w3.org/2000/svg',name);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);if(text!=null)el.textContent=text;return el;}
function chart(rows){
  const svg=$('chart');svg.replaceChildren();
  if(!rows.length){svg.append(node('text',{x:420,y:120,'text-anchor':'middle'},'Chưa có phiên quan sát'));return;}
  const max=Math.max(1,...rows.map(x=>x.usage_kWh))*1.1, left=45,right=825,top=18,bottom=208;
  const x=i=>left+i*(right-left)/(rows.length-1), y=v=>bottom-v/max*(bottom-top);
  for(let i=0;i<=4;i++){const yy=y(max*i/4);svg.append(node('line',{x1:left,y1:yy,x2:right,y2:yy,class:'grid'}));svg.append(node('text',{x:left-8,y:yy+4,'text-anchor':'end'},Math.round(max*i/4)));}
  const points=rows.map((r,i)=>`${x(i)},${y(r.usage_kWh)}`).join(' ');
  svg.append(node('polygon',{points:`${left},${bottom} ${points} ${right},${bottom}`,class:'area'}));
  svg.append(node('polyline',{points,class:'series'}));
  for(const i of [0,Math.floor(rows.length/3),Math.floor(2*rows.length/3),rows.length-1])svg.append(node('text',{x:x(i),y:235,'text-anchor':i===0?'start':i===rows.length-1?'end':'middle'},timestamp(rows[i].time)));
}
function render(data){
  if(current?.session !== data.session || current?.clock !== data.clock){
    $('decisionNote').value=''; $('decisionFeedback').textContent=''; decisionToken=null;
  }
  current=data; $('clock').textContent=data.clock?`${timestamp(data.clock)} / 2018`:'Chưa mở phiên';
  $('start').min=data.first_start.slice(0,16); $('start').max=data.last_start.slice(0,16);
  if($('start').value < $('start').min || $('start').value > $('start').max) $('start').value=$('start').min;
  $('periodNote').textContent=`Phát lại từ ${timestamp(data.first_start)} đến ${timestamp(data.end)} / 2018. Actual chỉ xuất hiện sau đủ 60 phút và đủ bốn bản đo.`;
  $('sessionNote').textContent=data.session?'Đồng hồ dữ liệu · mỗi bước 15 phút':'Chọn một thời điểm để bắt đầu.';
  const response=data.current_forecast, forecast=response?.prediction;
  $('prediction').textContent=forecast?`${number(forecast.predicted_next_60m_kWh)} kWh`:'—';
  $('horizon').textContent=forecast?`${timestamp(forecast.forecast_start)} → ${timestamp(forecast.forecast_end)}`:data.finished?'Đã kết thúc phiên dữ liệu':response?.detail?'Chưa có dự báo hợp lệ':'Chưa có dự báo';
  $('quality').textContent=data.finished?'Hoàn tất phát lại':response?statuses[response.status]||response.status:'Chưa có phiên';
  const policy=data.policy_info, decision=response?.decision;
  $('alertHeading').textContent=data.alert_policy==='configured'?'Policy nghiên cứu đã hiệu chỉnh':data.alert_policy==='policy_unavailable'?'Policy không hợp lệ':'Chưa có policy cảnh báo';
  $('alertDescription').textContent=policy?.threshold_kWh!=null?`Ngưỡng ${number(policy.threshold_kWh)} kWh · Đệm ${number(policy.upper_buffer_kWh)} kWh. Có hiệu lực từ ${timestamp(policy.valid_from)}. Đây không phải giới hạn công suất hợp đồng.`:policy?.detail||'Chỉ dự báo; chưa kết luận vượt ngưỡng hoặc tải an toàn.';
  $('alertStatus').textContent=decision?.alert===true?'Đề nghị kiểm tra khoảng tiêu thụ cao':decision?.alert===false?'Dưới ngưỡng nghiên cứu':'Chưa có cảnh báo hợp lệ';
  $('recommendation').textContent=decision?.recommendation||'Mở phiên để xem dự báo và khuyến nghị. Người phụ trách xác nhận; không gửi lệnh tới máy.';
  const info=data.model_info;
  $('modelInfo').textContent=info?`${info.model_name} · ${info.run_directory}\nMAE validation: ${number(info.validation_mae_kWh)} kWh; RMSE: ${number(info.validation_rmse_kWh)} kWh.\nFeature: ${info.feature_implementation}\nSHA-256: ${info.model_sha256}`:'Thông tin sẽ lấy từ model nạp hợp lệ khi mở phiên; không dùng điểm của model cũ.';
  const m=data.metrics?.models?.[0];
  $('mae').textContent=m?.all_matured.rows?`${number(m.all_matured.mae_kWh)} kWh`:'—';
  $('scored').textContent=m?`${m.all_matured.rows} dự báo đã có đủ số đo thực tế`:'Chưa có kết quả đối chiếu';
  $('pending').textContent=m?m.pending:'—';$('total').textContent=`${data.metrics?.forecasts||0} dự báo`;
  $('rollingMae').textContent=m?.last_24h_by_forecast_end.rows?`${number(m.last_24h_by_forecast_end.mae_kWh)} kWh`:'—';
  $('rollingBias').textContent=m?.last_24h_by_forecast_end.rows?`${number(m.last_24h_by_forecast_end.bias_kWh)} kWh`:'—';
  $('rollingCount').textContent=m?m.last_24h_by_forecast_end.rows:'—';
  const quality=data.metrics?.input_quality;
  $('invalidRate').textContent=quality?.forecast_attempts?`${quality.invalid_history_attempts}/${quality.forecast_attempts} (${number(100*quality.invalid_history_rate)}%)`:'—';
  const usageWatch=response?.usage_watch, performanceWatch=data.metrics?.performance_watch;
  $('reviewStatus').textContent=performanceWatch?.status==='needs_review' || usageWatch?.status==='needs_review'?'Cần rà soát dữ liệu và hiệu năng':'Chưa có tín hiệu rà soát / chưa đủ nhãn';
  $('reviewDetail').textContent=`PSI 7 ngày: ${number(usageWatch?.psi)}. Nhãn đủ cho kiểm hai tuần: ${performanceWatch?.available_labels||0}/${performanceWatch?.required_labels||1344}. Các ngưỡng là heuristic, không chẩn đoán drift hoặc tự retrain.`;
  $('decisionContext').textContent=response?`Quan sát lúc ${timestamp(data.clock)} · ${statuses[response.status]||response.status}. Dừng phát lại trước khi ghi nhận.`:'Không có quan sát hiện tại để ghi nhận.';
  $('decisions').replaceChildren();
  for(const d of data.decisions){
    const li=document.createElement('li');
    const saved=d.context.decision;
    const savedAlert=saved?.alert===true?'Đề nghị kiểm tra':saved?.alert===false?'Dưới ngưỡng nghiên cứu':'Không có cảnh báo hợp lệ';
    li.textContent=`${timestamp(d.replay_time)} · ${d.operator} · ${decisionLabels[d.action]}\n${d.note}\nTrạng thái khi ghi: ${statuses[d.context.status]||d.context.status} · Dự báo: ${number(d.context.predicted_kWh)} kWh · ${savedAlert}`;
    $('decisions').append(li);
  }
  if(!data.decisions.length){const li=document.createElement('li');li.textContent='Chưa có ghi nhận trong phiên.';$('decisions').append(li);}
  chart(data.history);
  $('rows').replaceChildren();
  for(const r of data.forecasts){const tr=document.createElement('tr');const label=r.label_status==='waiting_for_observations'?'Đã đủ giờ · thiếu bản đo':r.actual==null?'Chờ đủ 60 phút':'Đã đối chiếu';for(const v of [timestamp(r.issue_time),timestamp(r.forecast_end),number(r.prediction),number(r.actual),number(r.error),r.decision?.alert===true?'Kiểm tra':r.decision?.alert===false?'Dưới ngưỡng':'—',label]){const td=document.createElement('td');td.textContent=v;tr.append(td);}tr.lastChild.className=r.actual==null?'waiting':'available';$('rows').append(tr);}
  if(!data.forecasts.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=7;td.className='empty';td.textContent='Chưa có dự báo. Mở phiên hoặc kiểm tra trạng thái xử lý.';tr.append(td);$('rows').append(tr);}
  const p=data.performance;
  $('ram').textContent=p?.process_rss_bytes!=null?`${number(p.process_rss_bytes/1048576)} MB`:'—';
  $('cpu').textContent=p?`${number(p.process_cpu_ms)} ms`:'—';$('backend').textContent=p?`${number(p.backend_ms)} ms`:'—';
  $('p95').textContent=p?.session_backend_p95_ms!=null?`${number(p.session_backend_p95_ms)} ms (${p.timed_actions} bước)`:'—';
  $('modelsize').textContent=p?.model_bytes!=null?`${number(p.model_bytes/1024)} KB`:'—';
  $('errors').replaceChildren();
  for(const e of data.errors){const li=document.createElement('li');li.textContent=`${timestamp(e.issue_time)} · ${statuses[e.status]||e.status}: ${e.detail}`;$('errors').append(li);}
  if(!data.errors.length){const li=document.createElement('li');li.textContent='Chưa ghi nhận lỗi trong phiên.';$('errors').append(li);}
  if(data.finished || (response && response.status!=='forecast_ready')) stop();controls();
}
async function request(path,body){
  if(busy)return;busy=true;controls();$('error').hidden=true;const begin=performance.now();
  try{
    const response=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});
    const data=await response.json();if(!response.ok)throw new Error(data.error||'Không kết nối được server.');
    render(data);await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
    const elapsed=performance.now()-begin;$('browser').textContent=`${number(elapsed)} ms`;
    window.lastRenderMeasurement={request_to_render_ms:elapsed,clock:data.clock,backend_ms:data.performance.backend_ms,process_cpu_ms:data.performance.process_cpu_ms,process_rss_bytes:data.performance.process_rss_bytes};
    return true;
  }catch(error){stop();$('error').textContent=error.message;$('error').hidden=false;return false;}
  finally{busy=false;controls();}
}
async function tick(){if(!playing)return;await request('/api/step',{});if(playing)timer=setTimeout(tick,Number($('speed').value));}
$('reset').addEventListener('click',async()=>{stop();if(!$('start').checkValidity()){$('start').reportValidity();return;}await request('/api/reset',{start:$('start').value});});
$('step').addEventListener('click',()=>request('/api/step',{}));
$('play').addEventListener('click',()=>{if(playing){stop();return;}playing=true;controls();tick();});
for(const id of ['operator','decisionNote']) $(id).addEventListener('input',()=>{decisionToken=null;$('decisionFeedback').textContent='';});
for(const action of Object.keys(decisionLabels)) $(action).addEventListener('click',async()=>{
  if(busy || playing || !current?.current_forecast)return;
  const operator=$('operator').value.trim(),note=$('decisionNote').value.trim();
  if(!operator || !note){$('decisionFeedback').textContent='Nhập tên người ghi và ghi chú/lý do trước khi lưu.';return;}
  if(!decisionToken || decisionToken.action!==action)decisionToken={action,token:crypto.randomUUID()};
  const saved=await request('/api/decision',{session:current.session,clock:current.clock,request_id:current.current_forecast.request_id,
    operator,action,note,token:decisionToken.token});
  if(saved){$('decisionNote').value='';decisionToken=null;$('decisionFeedback').textContent='Đã lưu ghi nhận trong nhật ký phiên. Không thực hiện thao tác trên thiết bị.';}
});
request('/api/state');
