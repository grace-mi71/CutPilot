"use client";
import {useEffect,useRef,useState} from "react";
import EditorHeader from "./EditorHeader";
import VideoPreview from "./VideoPreview";
import WorkList from "./WorkList";
import ChatPanel from "./ChatPanel";
import {initialMessages,initialWorks} from "@/data/mockData";
import type {Message,WorkItem} from "@/types/editor";
export default function Editor(){
 const [messages,setMessages]=useState<Message[]>(initialMessages); const [works,setWorks]=useState<WorkItem[]>(initialWorks); const [input,setInput]=useState(""); const [videoUrl,setVideoUrl]=useState<string|null>(null);
 const [selectPrompt,setSelectPrompt]=useState(false);
 const [selecting,setSelecting]=useState(false);
 const [view,setView]=useState<"original" | "complete">("original");
 const promptTimer=useRef<ReturnType<typeof setTimeout> | null>(null);
 useEffect(()=>()=>{if(videoUrl)URL.revokeObjectURL(videoUrl)},[videoUrl]);
 useEffect(()=>()=>{if(promptTimer.current)clearTimeout(promptTimer.current)},[]);
 function select(file:File){setVideoUrl(URL.createObjectURL(file));cancelSelection()}
 function send(){const text=input.trim();if(!text)return;setMessages(old=>[...old,{id:Date.now(),sender:"user",text},{id:Date.now()+1,sender:"assistant",text:"아직 AI가 연결되지 않았어요. 지금은 화면을 확인하는 예시 모드입니다."}]);setInput("")}
 function add(){if(promptTimer.current)clearTimeout(promptTimer.current);setSelecting(true);setSelectPrompt(true);promptTimer.current=setTimeout(()=>setSelectPrompt(false),3000)}
 function cancelSelection(){if(promptTimer.current)clearTimeout(promptTimer.current);setSelecting(false);setSelectPrompt(false)}
 function changeView(next:"original" | "complete"){if(next==="complete")cancelSelection();setView(next)}
 function remove(id:number){setWorks(old=>old.filter(work=>work.id!==id))}
 return <main><EditorHeader/><div className="workspace"><div className={`editor ${view==="complete"?"editor-complete":""}`}><VideoPreview videoUrl={videoUrl} onSelect={select} selectPrompt={selectPrompt} selecting={selecting} onCancelSelection={cancelSelection} view={view} onViewChange={changeView}/><WorkList works={works} onAdd={add} onRemove={remove} readOnly={view==="complete"}/></div><ChatPanel messages={messages} input={input} setInput={setInput} onSend={send}/></div></main>
}
