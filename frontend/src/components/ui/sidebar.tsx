'use client'

import { cn } from "@/lib/utils"
import { LucideIcon, ChevronLeft } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Logo } from "@/components/icons"

interface SidebarItem {
  title: string
  href: string
  icon: LucideIcon
}

interface SidebarProps {
  items: SidebarItem[]
  className?: string
  onCollapse?: () => void
}

export function Sidebar({ items, className, onCollapse }: SidebarProps) {
  const pathname = usePathname()

  return (
    <aside className={cn("w-64 min-h-screen bg-white border-r border-gray-200 relative transition-all duration-300", className)}>
      <div className="p-6">
        <div className="flex items-center gap-3 mb-8">
          <div className="flex-shrink-0 flex items-center justify-center">
            <Logo size={36} color="#111827" />
          </div>
          <div className="flex flex-col justify-center gap-1 leading-none">
            <span className="text-xl font-bold text-foreground leading-none">
              DocScanner.ai
            </span>
            <span className="text-xs text-muted-foreground font-medium leading-none">
              AI 계약서 분석
            </span>
          </div>
        </div>

        <nav className="space-y-1">
          {items.map((item) => {
            const isActive = pathname === item.href
            const Icon = item.icon

            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200",
                  isActive
                    ? "bg-primary text-white shadow-sm"
                    : "text-slate-600 hover:text-slate-900 hover:bg-gray-50"
                )}
              >
                <Icon size={20} />
                <span>{item.title}</span>
              </Link>
            )
          })}
        </nav>
      </div>

      {onCollapse && (
        <button
          onClick={onCollapse}
          className="absolute top-1/2 -right-3 transform -translate-y-1/2 w-6 h-12 bg-white border border-gray-200 rounded-r-lg shadow-sm hover:bg-gray-50 transition-all duration-200 flex items-center justify-center group"
        >
          <ChevronLeft className="h-4 w-4 text-gray-400 group-hover:text-gray-600 transition-colors" />
        </button>
      )}
    </aside>
  )
}
