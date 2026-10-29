"use client";
import {useEffect,useRef,useState} from "react";
import EditorHeader from "./EditorHeader";
import VideoPreview from "./VideoPreview";
import WorkList from "./WorkList";
import ChatPanel from "./ChatPanel";
import {initialMessages,initialWorks} from "@/data/mockData";
import type {Message,OfflineConfirmation,WorkItem} from "@/types/editor";
import {createOfflineSession, OfflineChatError, sendOfflineMessage} from "@/lib/offlineChat";
export default function Editor(){
 const [messages,setMessages]=useState<Message[]>(initialMessages); const [works,setWorks]=useState<WorkItem[]>(initialWorks); const [input,setInput]=useState(""); const [videoUrl,setVideoUrl]=useState<string|null>(null);
 const [selectPrompt,setSelectPrompt]=useState(false);
 const [selecting,setSelecting]=useState(false);
 const [view,setView]=useState<"original" | "complete">("original");
 const [sending,setSending]=useState(false);
 const [pendingPlan,setPendingPlan]=useState<OfflineConfirmation | null>(null);
 const sendingRef=useRef(false);
 const sessionRef=useRef<string | null>(null);
 const messageIdRef=useRef(1);
 const promptTimer=useRef<ReturnType<typeof setTimeout> | null>(null);
 useEffect(()=>()=>{if(videoUrl)URL.revokeObjectURL(videoUrl)},[videoUrl]);
 useEffect(()=>()=>{if(promptTimer.current)clearTimeout(promptTimer.current)},[]);
 function select(file:File){setVideoUrl(URL.createObjectURL(file));cancelSelection()}
 async function send(){
  const text=input.trim();if(!text || sendingRef.current)return;
  sendingRef.current=true;setSending(true);setInput("");setPendingPlan(null);
  setMessages(old=>[...old,{id:++messageIdRef.current,sender:"user",text}]);
  try {
   if(!sessionRef.current)sessionRef.current=(await createOfflineSession()).id;
   const reply=await sendOfflineMessage(sessionRef.current,text);
   const notice=reply.blockers.length?"\n\n현재 실제 영상의 대상 선택은 연결되지 않았습니다. 가상 객체 person_01 또는 person_02로 요청해 보세요.":"";
   setMessages(old=>[...old,{id:++messageIdRef.current,sender:"assistant",text:reply.text+notice}]);
   if(reply.status==="plan" && reply.confirmation && reply.confirmation.plan.operations.length>0 && reply.blockers.length===0 && !reply.confirmation.items.some(item=>item.status==="choose")){
    setPendingPlan(reply.confirmation);
   }
  } catch(error) {
   if(error instanceof OfflineChatError && error.status===404)sessionRef.current=null;
   const message=error instanceof OfflineChatError?error.message:"오류가 발생했습니다. 잠시 후 다시 시도해 주세요.";
   setMessages(old=>[...old,{id:++messageIdRef.current,sender:"assistant",text:message}]);
   setInput(current=>current || text);
  } finally {sendingRef.current=false;setSending(false)}
 }
 function add(){if(promptTimer.current)clearTimeout(promptTimer.current);setSelecting(true);setSelectPrompt(true);promptTimer.current=setTimeout(()=>setSelectPrompt(false),3000)}
 function cancelSelection(){if(promptTimer.current)clearTimeout(promptTimer.current);setSelecting(false);setSelectPrompt(false)}
 function changeView(next:"original" | "complete"){if(next==="complete")cancelSelection();setView(next)}
 function remove(id:number){setWorks(old=>old.filter(work=>work.id!==id))}
 function confirmPlan(){
  if(!pendingPlan)return;
  const items=pendingPlan.items;
  const label=(index:number,field:"op"|"selector"|"time_range")=>items.find(item=>item.operation_index===index && item.field===field)?.label || "";
  setWorks(pendingPlan.plan.operations.map((op,index)=>({id:index,object:label(index,"selector") || "전체",time:label(index,"time_range") || "전체 구간",action:label(index,"op") || op.op})));
  setPendingPlan(null);
  setMessages(old=>[...old,{id:++messageIdRef.current,sender:"assistant",text:"확인한 편집 계획을 작업 리스트에 반영했습니다."}]);
 }
 return <main><EditorHeader/><div className="workspace"><div className={`editor ${view==="complete"?"editor-complete":""}`}><VideoPreview videoUrl={videoUrl} onSelect={select} selectPrompt={selectPrompt} selecting={selecting} onCancelSelection={cancelSelection} view={view} onViewChange={changeView}/><WorkList works={works} onAdd={add} onRemove={remove} readOnly={view==="complete"}/></div><ChatPanel messages={messages} input={input} setInput={setInput} onSend={send} sending={sending} hasPendingPlan={pendingPlan!==null} onConfirmPlan={confirmPlan}/></div></main>
}
