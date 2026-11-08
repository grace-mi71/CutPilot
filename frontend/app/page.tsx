import { Suspense } from "react";
import Editor from "@/components/Editor";

type SearchParams = Promise<{ [key: string]: string | string[] | undefined }>;

// ?mode=text hides the boxes on the preview: the "text only" condition of the user study.
async function EditorForMode({ searchParams }: { searchParams: SearchParams }) {
  const { mode } = await searchParams;
  return <Editor textOnly={mode === "text"} />;
}

export default function Home({ searchParams }: { searchParams: SearchParams }) {
  return (
    <Suspense fallback={null}>
      <EditorForMode searchParams={searchParams} />
    </Suspense>
  );
}
