import type { Metadata } from "next"
import { Geist, Geist_Mono, Bebas_Neue } from "next/font/google"  // ← add

import "./globals.css"

import { Footer } from "@/components/layout/footer"

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
})

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
})

const bebasNeue = Bebas_Neue({   // ← add
  variable: "--font-bebas",
  subsets: ["latin"],
  weight: "400",
})

export const metadata: Metadata = {
  title: "Blog Agent OS",
  description: "Autonomous LangGraph workspace for AI blog generation.",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${bebasNeue.variable} h-full antialiased`}  // ← add variable
    >
      <body className="min-h-screen bg-black text-white">
        <main className=" min-h-screen">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  )
}