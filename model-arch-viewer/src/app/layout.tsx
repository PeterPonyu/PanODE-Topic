import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PanODE Model Architecture Viewer",
  description: "Interactive architecture diagrams for PanODE-LAB model variants",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-white text-gray-900 antialiased">{children}</body>
    </html>
  );
}
