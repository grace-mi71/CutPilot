import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "CutPilot | 영상 편집", description: "대화로 편집 계획을 만드는 CutPilot" };
export default function RootLayout({children}: Readonly<{children: React.ReactNode}>) { return <html lang="ko"><body>{children}</body></html>; }
