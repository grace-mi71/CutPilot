import type { Message } from "@/types/editor";

export const initialMessages: Message[] = [
  {
    id: 1,
    sender: "assistant",
    text:
      "영상을 올린 뒤 편집 계획을 말씀해 주세요. 여러 단계도 됩니다.\n" +
      "예) 1. 검은 모자 쓴 남자 모자이크  2. 빨간 코트 입은 사람 블러  3. 세로로 바꿔줘\n" +
      "진행하다 확인이 필요한 것만 하나씩 여쭤볼게요.",
  },
];
