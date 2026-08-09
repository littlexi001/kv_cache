import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Attention Confidence Lab · Qwen3-8B",
  description: "Clean two-hop long-context attention and answer-confidence explorer.",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
