import type { Message, WorkItem } from "@/types/editor";
export const initialMessages: Message[] = [{ id: 1, sender: "assistant", text: "어떤 영상 편집을 도와드릴까요?" }];
export const initialWorks: WorkItem[] = [{id:1,object:"Person 01",time:"00:12 - 00:28",action:"블러"},{id:2,object:"Person 02",time:"00:03 - 00:05",action:"모자이크"}];
