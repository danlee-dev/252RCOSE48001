import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DocScanner AI",
  description: "AI 계약서 검토 서비스",
  icons: {
    icon: "/icon.svg",
    shortcut: "/icon.svg",
    apple: "/icon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
