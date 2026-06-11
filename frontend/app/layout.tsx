import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title:       "Retail Purchase Predictor",
  description: "Real-time webcam-driven purchase intent prediction",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
