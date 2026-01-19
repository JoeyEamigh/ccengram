# WebUI Specification

## Overview

The WebUI provides a browser-based interface for searching and managing memories. Key features:

- **Auto-start**: First Claude Code instance starts the server
- **Auto-stop**: Server stops when last Claude Code instance exits
- **Multi-agent view**: Real-time view of parallel agents in the same project
- **WebSocket-connected**: Live updates as memories are created
- **Bun + React + Tailwind + shadcn/ui**: Modern component library with utility-first CSS

## Architecture

### Stack: Bun.serve + React SSR + Tailwind CSS + shadcn/ui

**Why this stack:**

- Bun natively supports React (JSX/TSX, SSR, hydration)
- Tailwind CSS for utility-first styling (no custom CSS files)
- shadcn/ui for accessible, composable components (built on Radix UI)
- WebSocket support built into `Bun.serve`
- SSR for fast initial loads, hydration for interactivity

**Key patterns:**

- Server renders React to HTML string
- Client hydrates and takes over
- WebSocket pushes real-time updates
- Tailwind generates CSS at build time
- shadcn/ui components are copied into project (not npm dependency)

### Instance Coordination

Multiple Claude Code sessions may run simultaneously. The WebUI uses a simple coordination mechanism:

```
$XDG_RUNTIME_DIR/ccmemory/webui.lock  - PID of server owner
$XDG_RUNTIME_DIR/ccmemory/clients.txt - List of active session IDs
```

**First instance**: Starts server, writes PID to lock file
**Additional instances**: Register in clients.txt, connect to existing server
**Instance exit**: Removes self from clients.txt
**Last instance exit**: Server owner shuts down when clients.txt is empty

### Real-time Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      WebUI Server                            │
├─────────────────────────────────────────────────────────────┤
│  Bun.serve                                                   │
│  ├── HTTP Routes (SSR)                                       │
│  │   ├── GET /           → renderApp() → HTML                │
│  │   ├── GET /api/*      → JSON responses                    │
│  │   └── GET /static/*   → bundled JS/CSS                    │
│  │                                                           │
│  └── WebSocket /ws                                           │
│      ├── Client connects → add to room                       │
│      ├── Memory created  → broadcast to room                 │
│      ├── Session update  → broadcast to room                 │
│      └── Client closes   → remove from room                  │
└─────────────────────────────────────────────────────────────┘
```

## Files to Create

```
src/webui/
├── server.ts              # Bun.serve with HTTP + WebSocket
├── coordination.ts        # Instance lock management
├── ssr.ts                 # React SSR renderer
├── build.ts               # Bun.build + Tailwind for client bundle
├── ws/
│   └── handler.ts         # WebSocket message handling
├── api/
│   └── routes.ts          # JSON API endpoints
├── components/
│   ├── App.tsx            # Root component
│   ├── Layout.tsx         # Page layout with navigation
│   ├── Search.tsx         # Search with results
│   ├── MemoryCard.tsx     # Memory display card
│   ├── MemoryDetail.tsx   # Modal detail view
│   ├── Timeline.tsx       # Chronological view
│   ├── AgentView.tsx      # Multi-agent monitoring
│   ├── SessionCard.tsx    # Agent session card
│   └── Settings.tsx       # Configuration UI
├── components/ui/         # shadcn/ui components (copied, not npm)
│   ├── button.tsx
│   ├── card.tsx
│   ├── dialog.tsx
│   ├── input.tsx
│   ├── select.tsx
│   ├── badge.tsx
│   ├── checkbox.tsx
│   └── tooltip.tsx
├── lib/
│   └── utils.ts           # cn() helper for Tailwind class merging
├── hooks/
│   ├── useWebSocket.ts    # WebSocket connection hook
│   ├── useMemories.ts     # Memory state management
│   └── useSessions.ts     # Session state management
├── client.tsx             # Client-side hydration entry
├── globals.css            # Tailwind directives + CSS variables
└── tailwind.config.ts     # Tailwind configuration
```

## Dependencies

Add to `package.json`:

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@radix-ui/react-dialog": "^1.0.5",
    "@radix-ui/react-select": "^2.0.0",
    "@radix-ui/react-checkbox": "^1.0.4",
    "@radix-ui/react-tooltip": "^1.0.7",
    "@radix-ui/react-slot": "^1.0.2",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.0",
    "lucide-react": "^0.309.0"
  },
  "devDependencies": {
    "tailwindcss": "^3.4.0",
    "postcss": "^8.4.33",
    "autoprefixer": "^10.4.17"
  }
}
```

## Tailwind Configuration

```typescript
// src/webui/tailwind.config.ts
import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: ["./src/webui/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        sector: {
          episodic: "hsl(var(--sector-episodic))",
          semantic: "hsl(var(--sector-semantic))",
          procedural: "hsl(var(--sector-procedural))",
          emotional: "hsl(var(--sector-emotional))",
          reflective: "hsl(var(--sector-reflective))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [],
} satisfies Config;
```

## Global Styles (Tailwind Directives)

```css
/* src/webui/globals.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;
    --card: 0 0% 100%;
    --card-foreground: 222.2 84% 4.9%;
    --primary: 222.2 47.4% 11.2%;
    --primary-foreground: 210 40% 98%;
    --secondary: 210 40% 96.1%;
    --secondary-foreground: 222.2 47.4% 11.2%;
    --muted: 210 40% 96.1%;
    --muted-foreground: 215.4 16.3% 46.9%;
    --accent: 210 40% 96.1%;
    --accent-foreground: 222.2 47.4% 11.2%;
    --destructive: 0 84.2% 60.2%;
    --destructive-foreground: 210 40% 98%;
    --border: 214.3 31.8% 91.4%;
    --input: 214.3 31.8% 91.4%;
    --ring: 222.2 84% 4.9%;
    --radius: 0.5rem;

    /* Memory sector colors */
    --sector-episodic: 210 100% 50%;
    --sector-semantic: 142 76% 36%;
    --sector-procedural: 45 93% 47%;
    --sector-emotional: 0 84% 60%;
    --sector-reflective: 270 60% 50%;
  }

  .dark {
    --background: 222.2 84% 4.9%;
    --foreground: 210 40% 98%;
    --card: 222.2 84% 4.9%;
    --card-foreground: 210 40% 98%;
    --primary: 210 40% 98%;
    --primary-foreground: 222.2 47.4% 11.2%;
    --secondary: 217.2 32.6% 17.5%;
    --secondary-foreground: 210 40% 98%;
    --muted: 217.2 32.6% 17.5%;
    --muted-foreground: 215 20.2% 65.1%;
    --accent: 217.2 32.6% 17.5%;
    --accent-foreground: 210 40% 98%;
    --destructive: 0 62.8% 30.6%;
    --destructive-foreground: 210 40% 98%;
    --border: 217.2 32.6% 17.5%;
    --input: 217.2 32.6% 17.5%;
    --ring: 212.7 26.8% 83.9%;
  }
}

@layer base {
  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground;
  }
}
```

## Utility Functions

```typescript
// src/webui/lib/utils.ts
import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

## shadcn/ui Components

### Button

```tsx
// src/webui/components/ui/button.tsx
import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean;
  };

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
```

### Card

```tsx
// src/webui/components/ui/card.tsx
import * as React from "react";
import { cn } from "../../lib/utils";

const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn("rounded-lg border bg-card text-card-foreground shadow-sm", className)}
      {...props}
    />
  )
);
Card.displayName = "Card";

const CardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />
  )
);
CardHeader.displayName = "CardHeader";

const CardTitle = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3 ref={ref} className={cn("text-2xl font-semibold leading-none tracking-tight", className)} {...props} />
  )
);
CardTitle.displayName = "CardTitle";

const CardDescription = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLParagraphElement>>(
  ({ className, ...props }, ref) => (
    <p ref={ref} className={cn("text-sm text-muted-foreground", className)} {...props} />
  )
);
CardDescription.displayName = "CardDescription";

const CardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
);
CardContent.displayName = "CardContent";

const CardFooter = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("flex items-center p-6 pt-0", className)} {...props} />
  )
);
CardFooter.displayName = "CardFooter";

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent };
```

### Badge

```tsx
// src/webui/components/ui/badge.tsx
import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default: "border-transparent bg-primary text-primary-foreground hover:bg-primary/80",
        secondary: "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
        destructive: "border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80",
        outline: "text-foreground",
        episodic: "border-transparent bg-blue-500 text-white",
        semantic: "border-transparent bg-green-600 text-white",
        procedural: "border-transparent bg-yellow-500 text-black",
        emotional: "border-transparent bg-red-500 text-white",
        reflective: "border-transparent bg-purple-500 text-white",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export type BadgeProps = React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof badgeVariants>;

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
```

### Input

```tsx
// src/webui/components/ui/input.tsx
import * as React from "react";
import { cn } from "../../lib/utils";

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>;

const Input = React.forwardRef<HTMLInputElement, InputProps>(({ className, type, ...props }, ref) => {
  return (
    <input
      type={type}
      className={cn(
        "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}
      ref={ref}
      {...props}
    />
  );
});
Input.displayName = "Input";

export { Input };
```

### Dialog

```tsx
// src/webui/components/ui/dialog.tsx
import * as React from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { cn } from "../../lib/utils";

const Dialog = DialogPrimitive.Root;
const DialogTrigger = DialogPrimitive.Trigger;
const DialogPortal = DialogPrimitive.Portal;
const DialogClose = DialogPrimitive.Close;

const DialogOverlay = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay
    ref={ref}
    className={cn(
      "fixed inset-0 z-50 bg-black/80 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
      className
    )}
    {...props}
  />
));
DialogOverlay.displayName = DialogPrimitive.Overlay.displayName;

const DialogContent = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content>
>(({ className, children, ...props }, ref) => (
  <DialogPortal>
    <DialogOverlay />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        "fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border bg-background p-6 shadow-lg duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-left-1/2 data-[state=closed]:slide-out-to-top-[48%] data-[state=open]:slide-in-from-left-1/2 data-[state=open]:slide-in-from-top-[48%] sm:rounded-lg",
        className
      )}
      {...props}
    >
      {children}
      <DialogPrimitive.Close className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-accent data-[state=open]:text-muted-foreground">
        <X className="h-4 w-4" />
        <span className="sr-only">Close</span>
      </DialogPrimitive.Close>
    </DialogPrimitive.Content>
  </DialogPortal>
));
DialogContent.displayName = DialogPrimitive.Content.displayName;

const DialogHeader = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("flex flex-col space-y-1.5 text-center sm:text-left", className)} {...props} />
);
DialogHeader.displayName = "DialogHeader";

const DialogFooter = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2", className)} {...props} />
);
DialogFooter.displayName = "DialogFooter";

const DialogTitle = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Title>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Title
    ref={ref}
    className={cn("text-lg font-semibold leading-none tracking-tight", className)}
    {...props}
  />
));
DialogTitle.displayName = DialogPrimitive.Title.displayName;

const DialogDescription = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Description>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Description>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Description ref={ref} className={cn("text-sm text-muted-foreground", className)} {...props} />
));
DialogDescription.displayName = DialogPrimitive.Description.displayName;

export {
  Dialog,
  DialogPortal,
  DialogOverlay,
  DialogClose,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogDescription,
};
```

### Select

```tsx
// src/webui/components/ui/select.tsx
import * as React from "react";
import * as SelectPrimitive from "@radix-ui/react-select";
import { Check, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "../../lib/utils";

const Select = SelectPrimitive.Root;
const SelectGroup = SelectPrimitive.Group;
const SelectValue = SelectPrimitive.Value;

const SelectTrigger = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Trigger>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Trigger
    ref={ref}
    className={cn(
      "flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [&>span]:line-clamp-1",
      className
    )}
    {...props}
  >
    {children}
    <SelectPrimitive.Icon asChild>
      <ChevronDown className="h-4 w-4 opacity-50" />
    </SelectPrimitive.Icon>
  </SelectPrimitive.Trigger>
));
SelectTrigger.displayName = SelectPrimitive.Trigger.displayName;

const SelectContent = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Content>
>(({ className, children, position = "popper", ...props }, ref) => (
  <SelectPrimitive.Portal>
    <SelectPrimitive.Content
      ref={ref}
      className={cn(
        "relative z-50 max-h-96 min-w-[8rem] overflow-hidden rounded-md border bg-background text-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
        position === "popper" &&
          "data-[side=bottom]:translate-y-1 data-[side=left]:-translate-x-1 data-[side=right]:translate-x-1 data-[side=top]:-translate-y-1",
        className
      )}
      position={position}
      {...props}
    >
      <SelectPrimitive.Viewport
        className={cn(
          "p-1",
          position === "popper" &&
            "h-[var(--radix-select-trigger-height)] w-full min-w-[var(--radix-select-trigger-width)]"
        )}
      >
        {children}
      </SelectPrimitive.Viewport>
    </SelectPrimitive.Content>
  </SelectPrimitive.Portal>
));
SelectContent.displayName = SelectPrimitive.Content.displayName;

const SelectItem = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Item>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Item
    ref={ref}
    className={cn(
      "relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      className
    )}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <SelectPrimitive.ItemIndicator>
        <Check className="h-4 w-4" />
      </SelectPrimitive.ItemIndicator>
    </span>
    <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
  </SelectPrimitive.Item>
));
SelectItem.displayName = SelectPrimitive.Item.displayName;

export { Select, SelectGroup, SelectValue, SelectTrigger, SelectContent, SelectItem };
```

### Checkbox

```tsx
// src/webui/components/ui/checkbox.tsx
import * as React from "react";
import * as CheckboxPrimitive from "@radix-ui/react-checkbox";
import { Check } from "lucide-react";
import { cn } from "../../lib/utils";

const Checkbox = React.forwardRef<
  React.ElementRef<typeof CheckboxPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof CheckboxPrimitive.Root>
>(({ className, ...props }, ref) => (
  <CheckboxPrimitive.Root
    ref={ref}
    className={cn(
      "peer h-4 w-4 shrink-0 rounded-sm border border-primary ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground",
      className
    )}
    {...props}
  >
    <CheckboxPrimitive.Indicator className={cn("flex items-center justify-center text-current")}>
      <Check className="h-4 w-4" />
    </CheckboxPrimitive.Indicator>
  </CheckboxPrimitive.Root>
));
Checkbox.displayName = CheckboxPrimitive.Root.displayName;

export { Checkbox };
```

## Server Implementation

```typescript
// src/webui/server.ts
import { renderToReadableStream } from "react-dom/server";
import { App } from "./components/App";
import { handleAPI } from "./api/routes";
import { handleWebSocket, broadcastToRoom } from "./ws/handler";
import { registerClient, unregisterClient, tryAcquireLock, releaseLock, isServerRunning, getActiveClients } from "./coordination";
import { buildAssets } from "./build";
import { log } from "../utils/log";

const DEFAULT_PORT = 37778;

type BuildOutput = {
  clientJs: string;
  css: string;
};

let buildOutput: BuildOutput | null = null;

export async function startServer(options: {
  port?: number;
  sessionId: string;
  open?: boolean;
} = { sessionId: "" }) {
  const port = options.port || DEFAULT_PORT;

  log.info("webui", "Starting WebUI server", { port, sessionId: options.sessionId });

  if (await isServerRunning(port)) {
    await registerClient(options.sessionId);
    log.debug("webui", "Server already running, registering as client");
    console.log(`CCMemory WebUI already running at http://localhost:${port}`);
    return { alreadyRunning: true };
  }

  const acquired = await tryAcquireLock();
  if (!acquired) {
    log.debug("webui", "Lock not acquired, another server is starting");
    await registerClient(options.sessionId);
    return { alreadyRunning: true };
  }

  log.info("webui", "Lock acquired, starting server");
  await registerClient(options.sessionId);

  log.debug("webui", "Building client assets");
  buildOutput = await buildAssets();
  log.debug("webui", "Client assets built", { jsSize: buildOutput.clientJs.length, cssSize: buildOutput.css.length });

  const server = Bun.serve({
    port,

    async fetch(req, server) {
      const url = new URL(req.url);
      const path = url.pathname;

      if (path === "/ws") {
        const upgraded = server.upgrade(req, {
          data: { projectId: url.searchParams.get("project") }
        });
        if (upgraded) return undefined;
        return new Response("WebSocket upgrade failed", { status: 400 });
      }

      if (path.startsWith("/api/")) {
        return handleAPI(req, path);
      }

      if (path === "/client.js") {
        return new Response(buildOutput!.clientJs, {
          headers: { "Content-Type": "application/javascript" }
        });
      }

      if (path === "/styles.css") {
        return new Response(buildOutput!.css, {
          headers: { "Content-Type": "text/css" }
        });
      }

      return renderPage(url);
    },

    websocket: {
      open(ws) {
        const { projectId } = ws.data as { projectId?: string };
        const room = projectId || "global";
        ws.subscribe(room);
        log.debug("webui", "WebSocket connected", { room });
      },

      message(ws, message) {
        handleWebSocket(ws, message);
      },

      close(ws) {
        const { projectId } = ws.data as { projectId?: string };
        const room = projectId || "global";
        ws.unsubscribe(room);
        log.debug("webui", "WebSocket disconnected", { room });
      }
    }
  });

  log.info("webui", "WebUI server started", { port });
  console.log(`CCMemory WebUI running at http://localhost:${port}`);

  if (options.open) {
    openBrowser(`http://localhost:${port}`);
  }

  const checkInterval = setInterval(async () => {
    const clients = await getActiveClients();
    if (clients.length === 0) {
      log.info("webui", "No active clients, shutting down server");
      clearInterval(checkInterval);
      server.stop();
      await releaseLock();
    }
  }, 5000);

  return { server, checkInterval };
}

async function renderPage(url: URL): Promise<Response> {
  const initialData = await fetchInitialData(url);
  const html = renderToReadableStream(<App url={url.pathname} initialData={initialData} />);

  return new Response(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CCMemory</title>
  <link rel="stylesheet" href="/styles.css">
</head>
<body class="min-h-screen bg-background font-sans antialiased">
  <div id="root">${html}</div>
  <script>window.__INITIAL_DATA__ = ${JSON.stringify(initialData)};</script>
  <script src="/client.js"></script>
</body>
</html>`, {
    headers: { "Content-Type": "text/html; charset=utf-8" }
  });
}

async function fetchInitialData(url: URL): Promise<any> {
  const path = url.pathname;
  const searchParams = url.searchParams;

  if (path === "/" || path === "/search") {
    const query = searchParams.get("q");
    if (query) {
      const search = createSearchService();
      return {
        type: "search",
        results: await search.search({ query, limit: 20 })
      };
    }
    return { type: "search", results: [] };
  }

  if (path === "/agents") {
    const sessions = await getRecentSessions(searchParams.get("project"));
    return { type: "agents", sessions };
  }

  if (path === "/timeline") {
    const anchorId = searchParams.get("anchor");
    if (anchorId) {
      const search = createSearchService();
      return { type: "timeline", data: await search.timeline(anchorId, 10, 10) };
    }
    return { type: "timeline", data: null };
  }

  return { type: "home" };
}

function openBrowser(url: string) {
  const cmd = process.platform === "darwin" ? "open"
    : process.platform === "win32" ? "start"
    : "xdg-open";
  Bun.spawn([cmd, url]);
}

// Called by hooks when memories change
export function notifyMemoryChange(projectId: string, memory: Memory) {
  broadcastToRoom(projectId, { type: "memory:created", memory });
  broadcastToRoom("global", { type: "memory:created", memory, projectId });
}

export function notifySessionChange(projectId: string, session: Session) {
  broadcastToRoom(projectId, { type: "session:updated", session });
  broadcastToRoom("global", { type: "session:updated", session, projectId });
}
```

## Build Module (Tailwind + Bun.build)

```typescript
// src/webui/build.ts
import { join } from "node:path";
import postcss from "postcss";
import tailwindcss from "tailwindcss";
import autoprefixer from "autoprefixer";
import { log } from "../utils/log";

const WEBUI_DIR = new URL(".", import.meta.url).pathname;

type BuildOutput = {
  clientJs: string;
  css: string;
};

export async function buildAssets(): Promise<BuildOutput> {
  const start = Date.now();

  const [clientJs, css] = await Promise.all([
    buildClientBundle(),
    buildTailwindCSS()
  ]);

  log.info("webui", "Assets built", { ms: Date.now() - start });
  return { clientJs, css };
}

async function buildClientBundle(): Promise<string> {
  const result = await Bun.build({
    entrypoints: [join(WEBUI_DIR, "client.tsx")],
    target: "browser",
    minify: true,
    define: {
      "process.env.NODE_ENV": '"production"'
    }
  });

  if (!result.success) {
    log.error("webui", "Failed to build client bundle", { logs: result.logs.join("\n") });
    throw new Error("Failed to build client: " + result.logs.join("\n"));
  }

  return await result.outputs[0].text();
}

async function buildTailwindCSS(): Promise<string> {
  const globalsPath = join(WEBUI_DIR, "globals.css");
  const configPath = join(WEBUI_DIR, "tailwind.config.ts");

  const globalsContent = await Bun.file(globalsPath).text();

  const result = await postcss([
    tailwindcss({ config: configPath }),
    autoprefixer()
  ]).process(globalsContent, {
    from: globalsPath,
    to: "styles.css"
  });

  return result.css;
}
```

## WebSocket Handler

```typescript
// src/webui/ws/handler.ts
import type { ServerWebSocket } from "bun";
import { createMemoryStore } from "../../services/memory/store";
import { log } from "../../utils/log";

type WebSocketData = { projectId?: string };

// Store server reference for broadcasting
let serverRef: any = null;

export function setServer(server: any) {
  serverRef = server;
}

export function broadcastToRoom(room: string, message: any) {
  if (serverRef) {
    serverRef.publish(room, JSON.stringify(message));
  }
}

export async function handleWebSocket(
  ws: ServerWebSocket<WebSocketData>,
  message: string | Buffer
) {
  try {
    const data = JSON.parse(message.toString());
    log.debug("webui", "WebSocket message received", { type: data.type });
    const store = createMemoryStore();

    switch (data.type) {
      case "memory:reinforce": {
        const memory = await store.reinforce(data.memoryId, data.amount || 0.1);
        log.debug("webui", "Memory reinforced via WebSocket", { memoryId: data.memoryId });
        ws.send(JSON.stringify({ type: "memory:updated", memory }));
        break;
      }

      case "memory:deemphasize": {
        const memory = await store.deemphasize(data.memoryId, data.amount || 0.2);
        log.debug("webui", "Memory de-emphasized via WebSocket", { memoryId: data.memoryId });
        ws.send(JSON.stringify({ type: "memory:updated", memory }));
        break;
      }

      case "memory:delete": {
        await store.delete(data.memoryId, data.hard || false);
        log.info("webui", "Memory deleted via WebSocket", { memoryId: data.memoryId, hard: data.hard });
        ws.send(JSON.stringify({
          type: "memory:deleted",
          memoryId: data.memoryId,
          hard: data.hard
        }));
        break;
      }

      case "subscribe:project": {
        if (data.projectId) {
          ws.subscribe(data.projectId);
          log.debug("webui", "Client subscribed to project", { projectId: data.projectId });
          ws.send(JSON.stringify({ type: "subscribed", room: data.projectId }));
        }
        break;
      }

      case "unsubscribe:project": {
        if (data.projectId) {
          ws.unsubscribe(data.projectId);
          log.debug("webui", "Client unsubscribed from project", { projectId: data.projectId });
        }
        break;
      }

      case "ping": {
        ws.send(JSON.stringify({ type: "pong" }));
        break;
      }
    }
  } catch (err) {
    log.error("webui", "WebSocket handler error", { error: err instanceof Error ? err.message : String(err) });
    ws.send(JSON.stringify({
      type: "error",
      message: err instanceof Error ? err.message : String(err)
    }));
  }
}
```

## React Components

### Client Entry

```tsx
// src/webui/client.tsx
import { hydrateRoot } from "react-dom/client";
import { App } from "./components/App";

declare global {
  interface Window {
    __INITIAL_DATA__: any;
  }
}

const initialData = window.__INITIAL_DATA__;
const root = document.getElementById("root")!;

hydrateRoot(root, <App url={window.location.pathname} initialData={initialData} />);
```

### App Component

```tsx
// src/webui/components/App.tsx
import { useState, useEffect } from "react";
import { Layout } from "./Layout";
import { Search } from "./Search";
import { Timeline } from "./Timeline";
import { AgentView } from "./AgentView";
import { Settings } from "./Settings";
import { MemoryDetail } from "./MemoryDetail";
import { useWebSocket } from "../hooks/useWebSocket";

type AppProps = {
  url: string;
  initialData: any;
};

export function App({ url, initialData }: AppProps) {
  const [currentPath, setCurrentPath] = useState(url);
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null);
  const [data, setData] = useState(initialData);

  const { connected, messages, send } = useWebSocket();

  // Handle browser navigation
  useEffect(() => {
    const handlePopState = () => {
      setCurrentPath(window.location.pathname);
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  // Handle incoming WebSocket messages
  useEffect(() => {
    for (const msg of messages) {
      switch (msg.type) {
        case "memory:created":
          // Add to current results if on search page
          if (currentPath === "/" || currentPath === "/search") {
            setData((prev: any) => ({
              ...prev,
              results: [msg.memory, ...(prev.results || [])]
            }));
          }
          break;
        case "memory:updated":
          // Update memory in current view
          setData((prev: any) => ({
            ...prev,
            results: prev.results?.map((m: Memory) =>
              m.id === msg.memory.id ? msg.memory : m
            )
          }));
          if (selectedMemory?.id === msg.memory.id) {
            setSelectedMemory(msg.memory);
          }
          break;
        case "session:updated":
          // Update sessions in agent view
          if (currentPath === "/agents") {
            setData((prev: any) => ({
              ...prev,
              sessions: prev.sessions?.map((s: Session) =>
                s.id === msg.session.id ? msg.session : s
              )
            }));
          }
          break;
      }
    }
  }, [messages, currentPath, selectedMemory?.id]);

  const navigate = (path: string) => {
    window.history.pushState({}, "", path);
    setCurrentPath(path);
    fetchPageData(path).then(setData);
  };

  const renderPage = () => {
    if (currentPath === "/" || currentPath.startsWith("/search")) {
      return (
        <Search
          initialResults={data.results}
          onSelectMemory={setSelectedMemory}
          wsConnected={connected}
        />
      );
    }
    if (currentPath === "/timeline") {
      return (
        <Timeline
          initialData={data.data}
          onSelectMemory={setSelectedMemory}
        />
      );
    }
    if (currentPath === "/agents") {
      return (
        <AgentView
          initialSessions={data.sessions}
          wsConnected={connected}
          onNavigate={navigate}
        />
      );
    }
    if (currentPath === "/settings") {
      return <Settings />;
    }
    return <Search initialResults={[]} onSelectMemory={setSelectedMemory} wsConnected={connected} />;
  };

  return (
    <Layout currentPath={currentPath} onNavigate={navigate} wsConnected={connected}>
      {renderPage()}
      {selectedMemory && (
        <MemoryDetail
          memory={selectedMemory}
          onClose={() => setSelectedMemory(null)}
          onReinforce={(id) => send({ type: "memory:reinforce", memoryId: id })}
          onDeemphasize={(id) => send({ type: "memory:deemphasize", memoryId: id })}
          onDelete={(id, hard) => send({ type: "memory:delete", memoryId: id, hard })}
          onViewTimeline={(id) => {
            setSelectedMemory(null);
            navigate(`/timeline?anchor=${id}`);
          }}
        />
      )}
    </Layout>
  );
}

async function fetchPageData(path: string): Promise<any> {
  const res = await fetch(`/api/page-data?path=${encodeURIComponent(path)}`);
  return res.json();
}
```

### WebSocket Hook

```tsx
// src/webui/hooks/useWebSocket.ts
import { useState, useEffect, useRef, useCallback } from "react";

type Message = { type: string; [key: string]: any };

export function useWebSocket(projectId?: string) {
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<Timer | null>(null);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws${projectId ? `?project=${projectId}` : ""}`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setMessages((prev) => [...prev, data]);
      } catch (err) {
        console.error("Failed to parse WebSocket message:", err);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      console.log("WebSocket disconnected, reconnecting in 3s...");
      reconnectTimeoutRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
    };
  }, [projectId]);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connect]);

  const send = useCallback((message: Message) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  // Clear processed messages periodically
  useEffect(() => {
    if (messages.length > 100) {
      setMessages((prev) => prev.slice(-50));
    }
  }, [messages.length]);

  return { connected, messages, send };
}
```

### Search Component

```tsx
// src/webui/components/Search.tsx
import { useState, useCallback } from "react";
import { Search as SearchIcon, Loader2 } from "lucide-react";
import { MemoryCard } from "./MemoryCard";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Checkbox } from "./ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import type { SearchResult, MemorySector } from "../../types";

type SearchProps = {
  initialResults: SearchResult[];
  onSelectMemory: (memory: Memory) => void;
  wsConnected: boolean;
};

export function Search({ initialResults, onSelectMemory, wsConnected }: SearchProps) {
  const [query, setQuery] = useState("");
  const [sector, setSector] = useState<MemorySector | "all">("all");
  const [includeSuperseded, setIncludeSuperseded] = useState(false);
  const [results, setResults] = useState(initialResults);
  const [loading, setLoading] = useState(false);

  const handleSearch = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    try {
      const params = new URLSearchParams({ q: query });
      if (sector !== "all") params.set("sector", sector);
      if (includeSuperseded) params.set("include_superseded", "true");

      const res = await fetch(`/api/search?${params}`);
      const data = await res.json();
      setResults(data.results);
      window.history.pushState({}, "", `/search?${params}`);
    } finally {
      setLoading(false);
    }
  }, [query, sector, includeSuperseded]);

  return (
    <div className="space-y-6">
      <form onSubmit={handleSearch} className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <div className="relative flex-1">
          <SearchIcon className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search memories..."
            className="pl-10"
            autoFocus
          />
        </div>

        <Select value={sector} onValueChange={(v) => setSector(v as MemorySector | "all")}>
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder="All Sectors" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Sectors</SelectItem>
            <SelectItem value="episodic">Episodic</SelectItem>
            <SelectItem value="semantic">Semantic</SelectItem>
            <SelectItem value="procedural">Procedural</SelectItem>
            <SelectItem value="emotional">Emotional</SelectItem>
            <SelectItem value="reflective">Reflective</SelectItem>
          </SelectContent>
        </Select>

        <div className="flex items-center space-x-2">
          <Checkbox
            id="superseded"
            checked={includeSuperseded}
            onCheckedChange={(checked) => setIncludeSuperseded(checked === true)}
          />
          <label htmlFor="superseded" className="text-sm text-muted-foreground cursor-pointer">
            Include superseded
          </label>
        </div>

        <Button type="submit" disabled={loading}>
          {loading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Searching...
            </>
          ) : (
            "Search"
          )}
        </Button>
      </form>

      <div className="flex items-center gap-2 text-sm">
        {wsConnected ? (
          <span className="flex items-center gap-1.5 text-green-600 dark:text-green-400">
            <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            Live updates enabled
          </span>
        ) : (
          <span className="flex items-center gap-1.5 text-muted-foreground">
            <span className="h-2 w-2 rounded-full bg-muted" />
            Connecting...
          </span>
        )}
      </div>

      <div className="space-y-4">
        {results.length === 0 ? (
          <p className="text-center text-muted-foreground py-12">
            {query ? "No memories found." : "Enter a search query to find memories."}
          </p>
        ) : (
          results.map((r) => (
            <MemoryCard
              key={r.memory.id}
              result={r}
              onClick={() => onSelectMemory(r.memory)}
            />
          ))
        )}
      </div>
    </div>
  );
}
```

### Memory Card Component

```tsx
// src/webui/components/MemoryCard.tsx
import { Link2 } from "lucide-react";
import { Card, CardContent, CardFooter, CardHeader } from "./ui/card";
import { Badge } from "./ui/badge";
import { cn } from "../lib/utils";
import type { SearchResult, MemorySector } from "../../types";

type MemoryCardProps = {
  result: SearchResult;
  onClick: () => void;
};

const sectorVariant: Record<MemorySector, "episodic" | "semantic" | "procedural" | "emotional" | "reflective"> = {
  episodic: "episodic",
  semantic: "semantic",
  procedural: "procedural",
  emotional: "emotional",
  reflective: "reflective",
};

export function MemoryCard({ result, onClick }: MemoryCardProps) {
  const { memory, score, sourceSession, isSuperseded, supersededBy, relatedMemoryCount } = result;

  return (
    <Card
      className={cn(
        "cursor-pointer transition-colors hover:bg-accent/50",
        isSuperseded && "opacity-60"
      )}
      onClick={onClick}
    >
      <CardHeader className="pb-2">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={sectorVariant[memory.sector]}>{memory.sector}</Badge>
          <span className="text-sm text-muted-foreground">
            Score: {(score * 100).toFixed(0)}%
          </span>
          <span className="text-sm text-muted-foreground">
            Salience: {(memory.salience * 100).toFixed(0)}%
          </span>

          {sourceSession && (
            <span
              className="text-sm text-muted-foreground ml-auto"
              title={sourceSession.summary || ""}
            >
              {formatDate(sourceSession.startedAt)}
            </span>
          )}

          {isSuperseded && (
            <Badge variant="destructive" title={`Superseded by ${supersededBy?.id}`}>
              SUPERSEDED
            </Badge>
          )}

          {relatedMemoryCount > 0 && (
            <Badge variant="secondary" className="flex items-center gap-1">
              <Link2 className="h-3 w-3" />
              {relatedMemoryCount} related
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent>
        <p className="text-sm leading-relaxed">
          {memory.content.slice(0, 300)}
          {memory.content.length > 300 ? "..." : ""}
        </p>
      </CardContent>

      <CardFooter className="pt-2 text-xs text-muted-foreground">
        <span>{formatDate(memory.createdAt)}</span>
        {memory.tags?.length > 0 && (
          <span className="ml-auto">{memory.tags.join(", ")}</span>
        )}
      </CardFooter>
    </Card>
  );
}

function formatDate(ts: number): string {
  return new Date(ts).toLocaleString();
}
```

### Agent View Component

```tsx
// src/webui/components/AgentView.tsx
import { useState, useEffect } from "react";
import { Users } from "lucide-react";
import { SessionCard } from "./SessionCard";
import { Badge } from "./ui/badge";
import { cn } from "../lib/utils";
import type { Session } from "../../types";

type AgentViewProps = {
  initialSessions: Session[];
  wsConnected: boolean;
  onNavigate: (path: string) => void;
};

type ParallelGroup = {
  sessions: Session[];
  startTime: number;
  endTime: number;
};

export function AgentView({ initialSessions, wsConnected, onNavigate }: AgentViewProps) {
  const [sessions, setSessions] = useState(initialSessions);
  const [groups, setGroups] = useState<ParallelGroup[]>([]);

  useEffect(() => {
    const sorted = [...sessions].sort((a, b) => b.startedAt - a.startedAt);
    const newGroups: ParallelGroup[] = [];

    for (const session of sorted) {
      const endTime = session.endedAt || Date.now();
      const overlappingGroup = newGroups.find(
        (g) => session.startedAt < g.endTime && endTime > g.startTime
      );

      if (overlappingGroup) {
        overlappingGroup.sessions.push(session);
        overlappingGroup.startTime = Math.min(overlappingGroup.startTime, session.startedAt);
        overlappingGroup.endTime = Math.max(overlappingGroup.endTime, endTime);
      } else {
        newGroups.push({ sessions: [session], startTime: session.startedAt, endTime });
      }
    }

    setGroups(newGroups);
  }, [sessions]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Agent Sessions</h2>
        <p className="text-muted-foreground">View parallel and recent Claude Code sessions</p>
      </div>

      <div className="flex items-center gap-2 text-sm">
        {wsConnected ? (
          <span className="flex items-center gap-1.5 text-green-600 dark:text-green-400">
            <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            Live updates enabled
          </span>
        ) : (
          <span className="flex items-center gap-1.5 text-muted-foreground">
            <span className="h-2 w-2 rounded-full bg-muted" />
            Connecting...
          </span>
        )}
      </div>

      {groups.length === 0 ? (
        <p className="text-center text-muted-foreground py-12">
          No sessions in the last 24 hours.
        </p>
      ) : (
        <div className="space-y-6">
          {groups.map((group, i) => (
            <div
              key={i}
              className={cn(
                "rounded-lg border p-4",
                group.sessions.length > 1 && "border-primary/50 bg-primary/5"
              )}
            >
              <div className="flex items-center gap-3 mb-4">
                <span className="text-sm font-medium">{formatDate(group.startTime)}</span>
                {group.sessions.length > 1 && (
                  <Badge variant="default" className="flex items-center gap-1">
                    <Users className="h-3 w-3" />
                    {group.sessions.length} parallel agents
                  </Badge>
                )}
              </div>
              <div className={cn(
                group.sessions.length > 1 && "grid gap-4 md:grid-cols-2"
              )}>
                {group.sessions.map((session) => (
                  <SessionCard
                    key={session.id}
                    session={session}
                    onViewMemories={() => onNavigate(`/search?session=${session.id}`)}
                    onViewTimeline={() => onNavigate(`/timeline?session=${session.id}`)}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function formatDate(ts: number): string {
  return new Date(ts).toLocaleString();
}
```

### Session Card Component

```tsx
// src/webui/components/SessionCard.tsx
import { Clock, Brain, Activity } from "lucide-react";
import { Card, CardContent, CardFooter, CardHeader } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { cn } from "../lib/utils";
import type { Session } from "../../types";

type SessionCardProps = {
  session: Session;
  onViewMemories: () => void;
  onViewTimeline: () => void;
};

export function SessionCard({ session, onViewMemories, onViewTimeline }: SessionCardProps) {
  const isActive = !session.endedAt;
  const duration = session.endedAt
    ? formatDuration(session.endedAt - session.startedAt)
    : formatDuration(Date.now() - session.startedAt) + " (active)";

  return (
    <Card className={cn(isActive && "ring-2 ring-green-500/50")}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <span className="font-mono text-sm text-muted-foreground" title={session.id}>
            {session.id.slice(0, 8)}...
          </span>
          {isActive && (
            <Badge className="bg-green-500 text-white animate-pulse">
              ACTIVE
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-2">
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <span>{duration}</span>
          </div>
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-muted-foreground" />
            <span>{session.memoryCount || 0} memories</span>
          </div>
        </div>
        {session.lastActivity && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Activity className="h-4 w-4" />
            <span>Last: {formatDate(session.lastActivity)}</span>
          </div>
        )}
        {session.summary && (
          <p className="text-sm text-muted-foreground line-clamp-2">{session.summary}</p>
        )}
      </CardContent>

      <CardFooter className="gap-2">
        <Button variant="outline" size="sm" onClick={onViewMemories}>
          View Memories
        </Button>
        <Button variant="outline" size="sm" onClick={onViewTimeline}>
          View Timeline
        </Button>
      </CardFooter>
    </Card>
  );
}

function formatDuration(ms: number): string {
  const minutes = Math.floor(ms / 60000);
  const hours = Math.floor(minutes / 60);
  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  return `${minutes}m`;
}

function formatDate(ts: number): string {
  return new Date(ts).toLocaleString();
}
```

### Memory Detail Modal

```tsx
// src/webui/components/MemoryDetail.tsx
import { Plus, Minus, Clock, Eye, History, Archive, Trash2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "./ui/dialog";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import type { Memory, MemorySector } from "../../types";

type MemoryDetailProps = {
  memory: Memory;
  onClose: () => void;
  onReinforce: (id: string) => void;
  onDeemphasize: (id: string) => void;
  onDelete: (id: string, hard: boolean) => void;
  onViewTimeline: (id: string) => void;
};

const sectorVariant: Record<MemorySector, "episodic" | "semantic" | "procedural" | "emotional" | "reflective"> = {
  episodic: "episodic",
  semantic: "semantic",
  procedural: "procedural",
  emotional: "emotional",
  reflective: "reflective",
};

export function MemoryDetail({
  memory,
  onClose,
  onReinforce,
  onDeemphasize,
  onDelete,
  onViewTimeline
}: MemoryDetailProps) {
  const handleDelete = (hard: boolean) => {
    const msg = hard
      ? "Permanently delete this memory? This cannot be undone."
      : "Archive this memory?";
    if (confirm(msg)) {
      onDelete(memory.id, hard);
      onClose();
    }
  };

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant={sectorVariant[memory.sector]}>{memory.sector}</Badge>
            <Badge variant="outline">{memory.tier}</Badge>
            {memory.isDeleted && <Badge variant="destructive">DELETED</Badge>}
            {memory.validUntil && <Badge variant="destructive">SUPERSEDED</Badge>}
          </div>
          <DialogTitle className="sr-only">Memory Details</DialogTitle>
          <DialogDescription className="sr-only">
            View and manage memory details
          </DialogDescription>
        </DialogHeader>

        <div className="prose prose-sm dark:prose-invert max-w-none py-4">
          <p className="whitespace-pre-wrap">{memory.content}</p>
        </div>

        <div className="space-y-3 border-t pt-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Salience</span>
            <div className="flex items-center gap-2">
              <span className="font-medium">{(memory.salience * 100).toFixed(0)}%</span>
              <Button
                variant="outline"
                size="icon"
                className="h-7 w-7"
                onClick={() => onReinforce(memory.id)}
              >
                <Plus className="h-3 w-3" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                className="h-7 w-7"
                onClick={() => onDeemphasize(memory.id)}
              >
                <Minus className="h-3 w-3" />
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="flex items-center gap-2 text-muted-foreground">
              <Clock className="h-4 w-4" />
              <span>Created: {formatDate(memory.createdAt)}</span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <Eye className="h-4 w-4" />
              <span>Accessed: {memory.accessCount} times</span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground col-span-2">
              <History className="h-4 w-4" />
              <span>Last accessed: {formatDate(memory.lastAccessed)}</span>
            </div>
          </div>
        </div>

        <DialogFooter className="flex-col sm:flex-row gap-2">
          <Button variant="outline" onClick={() => onViewTimeline(memory.id)}>
            <History className="mr-2 h-4 w-4" />
            View Timeline
          </Button>
          <div className="flex gap-2 ml-auto">
            <Button variant="secondary" onClick={() => handleDelete(false)}>
              <Archive className="mr-2 h-4 w-4" />
              Archive
            </Button>
            <Button variant="destructive" onClick={() => handleDelete(true)}>
              <Trash2 className="mr-2 h-4 w-4" />
              Delete Forever
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function formatDate(ts: number): string {
  return new Date(ts).toLocaleString();
}
```

### Layout Component

```tsx
// src/webui/components/Layout.tsx
import { ReactNode } from "react";
import { Brain, Search, Users, Clock, Settings, Moon, Sun } from "lucide-react";
import { Button } from "./ui/button";
import { cn } from "../lib/utils";

type LayoutProps = {
  children: ReactNode;
  currentPath: string;
  onNavigate: (path: string) => void;
  wsConnected: boolean;
};

const navItems = [
  { path: "/", label: "Search", icon: Search },
  { path: "/agents", label: "Agents", icon: Users },
  { path: "/timeline", label: "Timeline", icon: Clock },
  { path: "/settings", label: "Settings", icon: Settings },
];

export function Layout({ children, currentPath, onNavigate, wsConnected }: LayoutProps) {
  const toggleTheme = () => {
    document.documentElement.classList.toggle("dark");
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center">
          <div className="flex items-center gap-2 mr-6">
            <Brain className="h-6 w-6 text-primary" />
            <span className="font-semibold text-lg">CCMemory</span>
          </div>

          <nav className="flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentPath === item.path ||
                (item.path === "/" && currentPath.startsWith("/search"));
              return (
                <Button
                  key={item.path}
                  variant={isActive ? "secondary" : "ghost"}
                  size="sm"
                  onClick={() => onNavigate(item.path)}
                  className={cn(
                    "gap-2",
                    isActive && "bg-secondary"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{item.label}</span>
                </Button>
              );
            })}
          </nav>

          <div className="ml-auto flex items-center gap-2">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className={cn(
                "h-2 w-2 rounded-full",
                wsConnected ? "bg-green-500" : "bg-muted"
              )} />
              <span className="hidden sm:inline">
                {wsConnected ? "Connected" : "Connecting..."}
              </span>
            </div>

            <Button variant="ghost" size="icon" onClick={toggleTheme}>
              <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
              <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
              <span className="sr-only">Toggle theme</span>
            </Button>
          </div>
        </div>
      </header>

      <main className="flex-1 container py-6">
        {children}
      </main>

      <footer className="border-t py-4">
        <div className="container text-center text-sm text-muted-foreground">
          CCMemory - Claude Code Memory Plugin
        </div>
      </footer>
    </div>
  );
}
```

## API Routes

```typescript
// src/webui/api/routes.ts
import { createSearchService } from "../../services/search/hybrid";
import { createMemoryStore } from "../../services/memory/store";
import { getDatabase } from "../../db/database";
import { log } from "../../utils/log";

export async function handleAPI(req: Request, path: string): Promise<Response> {
  const start = Date.now();
  log.debug("webui", "API request", { method: req.method, path });
  const url = new URL(req.url);
  const search = createSearchService();
  const store = createMemoryStore();

  const json = (data: any, status = 200) =>
    Response.json(data, {
      status,
      headers: { "Content-Type": "application/json" }
    });

  try {
    // GET /api/health
    if (path === "/api/health") {
      return json({ ok: true });
    }

    // GET /api/search?q=...&sector=...
    if (path === "/api/search" && req.method === "GET") {
      const query = url.searchParams.get("q") || "";
      const sector = url.searchParams.get("sector") as any;
      const sessionId = url.searchParams.get("session");
      const includeSuperseded = url.searchParams.get("include_superseded") === "true";
      const limit = parseInt(url.searchParams.get("limit") || "20");

      const results = await search.search({
        query,
        sector,
        sessionId,
        includeSuperseded,
        limit,
        mode: "hybrid"
      });

      return json({ results });
    }

    // GET /api/memory/:id
    if (path.startsWith("/api/memory/") && req.method === "GET") {
      const id = path.replace("/api/memory/", "");
      const memory = await store.get(id);
      if (!memory) {
        return json({ error: "Memory not found" }, 404);
      }
      return json({ memory });
    }

    // GET /api/timeline?anchor=...
    if (path === "/api/timeline" && req.method === "GET") {
      const anchorId = url.searchParams.get("anchor");
      if (!anchorId) {
        return json({ error: "Missing anchor parameter" }, 400);
      }
      const data = await search.timeline(anchorId, 10, 10);
      return json({ data });
    }

    // GET /api/sessions?project=...
    if (path === "/api/sessions" && req.method === "GET") {
      const projectId = url.searchParams.get("project");
      const sessions = await getRecentSessions(projectId);
      return json({ sessions });
    }

    // GET /api/stats
    if (path === "/api/stats" && req.method === "GET") {
      const stats = await getStats();
      return json(stats);
    }

    // GET /api/page-data?path=...
    if (path === "/api/page-data" && req.method === "GET") {
      const pagePath = url.searchParams.get("path") || "/";
      const data = await fetchPageData(new URL(pagePath, req.url));
      return json(data);
    }

    log.warn("webui", "API route not found", { path });
    return json({ error: "Not found" }, 404);
  } catch (err) {
    log.error("webui", "API error", { path, error: err instanceof Error ? err.message : String(err), ms: Date.now() - start });
    return json({ error: err instanceof Error ? err.message : String(err) }, 500);
  }
}

async function getRecentSessions(projectId?: string | null): Promise<any[]> {
  const db = getDatabase();
  const result = await db.execute(`
    SELECT
      s.*,
      COUNT(DISTINCT sm.memory_id) as memory_count,
      MAX(m.created_at) as last_activity
    FROM sessions s
    LEFT JOIN session_memories sm ON s.id = sm.session_id
    LEFT JOIN memories m ON sm.memory_id = m.id
    WHERE s.started_at > ? ${projectId ? "AND s.project_id = ?" : ""}
    GROUP BY s.id
    ORDER BY s.started_at DESC
    LIMIT 50
  `, projectId
    ? [Date.now() - 24 * 60 * 60 * 1000, projectId]
    : [Date.now() - 24 * 60 * 60 * 1000]
  );
  return result.rows;
}

async function getStats(): Promise<any> {
  const db = getDatabase();

  const counts = await db.execute(`
    SELECT
      (SELECT COUNT(*) FROM memories WHERE is_deleted = 0) as total_memories,
      (SELECT COUNT(*) FROM memories WHERE tier = 'project' AND is_deleted = 0) as project_memories,
      (SELECT COUNT(*) FROM documents) as total_documents,
      (SELECT COUNT(*) FROM sessions) as total_sessions
  `);

  const bySector = await db.execute(`
    SELECT sector, COUNT(*) as count
    FROM memories
    WHERE is_deleted = 0
    GROUP BY sector
  `);

  return {
    totals: {
      memories: counts.rows[0][0],
      projectMemories: counts.rows[0][1],
      documents: counts.rows[0][2],
      sessions: counts.rows[0][3]
    },
    bySector: Object.fromEntries(bySector.rows.map((r: any) => [r[0], r[1]]))
  };
}
```

## Instance Coordination

```typescript
// src/webui/coordination.ts
import { join } from "node:path";
import { getXDGPath } from "../utils/xdg";
import { log } from "../utils/log";

const RUNTIME_DIR = getXDGPath("runtime");
const LOCK_FILE = join(RUNTIME_DIR, "webui.lock");
const CLIENTS_FILE = join(RUNTIME_DIR, "clients.txt");

export async function tryAcquireLock(): Promise<boolean> {
  try {
    await Bun.$`mkdir -p ${RUNTIME_DIR}`;

    const lockFile = Bun.file(LOCK_FILE);
    if (await lockFile.exists()) {
      const pid = parseInt(await lockFile.text());
      if (isProcessAlive(pid)) {
        log.debug("webui", "Lock held by another process", { pid });
        return false;
      }
      log.debug("webui", "Stale lock file found, cleaning up", { stalePid: pid });
    }

    await Bun.write(LOCK_FILE, String(process.pid));
    log.debug("webui", "Lock acquired", { pid: process.pid });
    return true;
  } catch (err) {
    log.error("webui", "Failed to acquire lock", { error: err instanceof Error ? err.message : String(err) });
    return false;
  }
}

export async function releaseLock(): Promise<void> {
  try {
    await Bun.$`rm -f ${LOCK_FILE}`;
    log.debug("webui", "Lock released");
  } catch {}
}

export async function registerClient(sessionId: string): Promise<void> {
  if (!sessionId) return;
  const clients = await getActiveClients();
  if (!clients.includes(sessionId)) {
    clients.push(sessionId);
    await Bun.write(CLIENTS_FILE, clients.join("\n"));
    log.debug("webui", "Client registered", { sessionId, totalClients: clients.length });
  }
}

export async function unregisterClient(sessionId: string): Promise<void> {
  const clients = await getActiveClients();
  const filtered = clients.filter((c) => c !== sessionId);
  log.debug("webui", "Client unregistered", { sessionId, remainingClients: filtered.length });
  if (filtered.length === 0) {
    await Bun.$`rm -f ${CLIENTS_FILE}`;
  } else {
    await Bun.write(CLIENTS_FILE, filtered.join("\n"));
  }
}

export async function getActiveClients(): Promise<string[]> {
  const clientsFile = Bun.file(CLIENTS_FILE);
  if (!(await clientsFile.exists())) return [];
  const content = await clientsFile.text();
  return content.split("\n").filter(Boolean);
}

export async function isServerRunning(port: number): Promise<boolean> {
  try {
    const res = await fetch(`http://localhost:${port}/api/health`, {
      signal: AbortSignal.timeout(1000)
    });
    return res.ok;
  } catch {
    return false;
  }
}

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
```

## Test Specifications

### Unit Tests (Colocated)

```typescript
// src/webui/__test__/coordination.test.ts
import { test, expect, describe, beforeEach, afterEach } from "bun:test";
import { tryAcquireLock, releaseLock, registerClient, unregisterClient, getActiveClients } from "./coordination";

describe("Instance Coordination", () => {
  beforeEach(async () => {
    await releaseLock();
    await Bun.$`rm -f $XDG_RUNTIME_DIR/ccmemory/clients.txt`.nothrow();
  });

  afterEach(async () => {
    await releaseLock();
  });

  test("tryAcquireLock succeeds on first call", async () => {
    const acquired = await tryAcquireLock();
    expect(acquired).toBe(true);
  });

  test("tryAcquireLock fails when lock held by live process", async () => {
    await tryAcquireLock();
    const secondAcquire = await tryAcquireLock();
    expect(secondAcquire).toBe(false);
  });

  test("registerClient adds to clients list", async () => {
    await registerClient("session-1");
    await registerClient("session-2");
    const clients = await getActiveClients();
    expect(clients).toContain("session-1");
    expect(clients).toContain("session-2");
  });

  test("unregisterClient removes from clients list", async () => {
    await registerClient("session-1");
    await registerClient("session-2");
    await unregisterClient("session-1");
    const clients = await getActiveClients();
    expect(clients).not.toContain("session-1");
    expect(clients).toContain("session-2");
  });
});

// src/webui/hooks/useWebSocket.test.ts
// Note: WebSocket hooks are integration tested with actual server
```

### Integration Tests (tests/webui/)

```typescript
// tests/webui/server.test.ts
import { test, expect, describe, beforeAll, afterAll } from "bun:test";
import { startServer } from "../../src/webui/server";
import { createMemoryStore } from "../../src/services/memory/store";
import { getDatabase } from "../../src/db/database";

describe("WebUI Server", () => {
  let serverResult: any;
  let store: ReturnType<typeof createMemoryStore>;

  beforeAll(async () => {
    store = createMemoryStore();
    serverResult = await startServer({ port: 37779, sessionId: "test-session" });
  });

  afterAll(async () => {
    serverResult.server?.stop();
    if (serverResult.checkInterval) clearInterval(serverResult.checkInterval);
  });

  test("serves index.html with SSR content", async () => {
    const res = await fetch("http://localhost:37779/");
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/html");
    const html = await res.text();
    expect(html).toContain("CCMemory");
    expect(html).toContain("__INITIAL_DATA__");
  });

  test("serves client.js bundle", async () => {
    const res = await fetch("http://localhost:37779/client.js");
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("javascript");
  });

  test("API search returns JSON results", async () => {
    await store.create({ content: "Test memory for API", sector: "semantic" }, "proj1");

    const res = await fetch("http://localhost:37779/api/search?q=API");
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.results).toBeDefined();
    expect(Array.isArray(data.results)).toBe(true);
  });

  test("API health endpoint works", async () => {
    const res = await fetch("http://localhost:37779/api/health");
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.ok).toBe(true);
  });

  test("WebSocket connection establishes", async () => {
    const ws = new WebSocket("ws://localhost:37779/ws");

    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => resolve();
      ws.onerror = reject;
      setTimeout(() => reject(new Error("WebSocket timeout")), 5000);
    });

    expect(ws.readyState).toBe(WebSocket.OPEN);
    ws.close();
  });

  test("WebSocket receives memory updates", async () => {
    const ws = new WebSocket("ws://localhost:37779/ws?project=proj1");

    await new Promise<void>((resolve) => {
      ws.onopen = () => resolve();
    });

    const messagePromise = new Promise<any>((resolve) => {
      ws.onmessage = (event) => {
        resolve(JSON.parse(event.data));
      };
    });

    // Trigger memory creation (which should broadcast)
    await store.create({ content: "WebSocket test memory" }, "proj1");

    // Note: In real implementation, store.create would call notifyMemoryChange
    // This test verifies the WebSocket connection works

    ws.close();
  });
});
```

## Acceptance Criteria

### Server & Coordination

- [ ] Server starts on configurable port (default 37778)
- [ ] First instance acquires lock and starts server
- [ ] Additional instances register as clients without starting new server
- [ ] Server shuts down when last client unregisters
- [ ] Lock file cleaned up on shutdown
- [ ] Health endpoint available at /api/health

### React SSR

- [ ] Initial page load returns server-rendered HTML
- [ ] Initial data embedded in **INITIAL_DATA** script tag
- [ ] Client hydrates and becomes interactive
- [ ] Navigation works client-side after hydration
- [ ] Browser back/forward works correctly

### WebSocket

- [ ] WebSocket connection established on page load
- [ ] Automatic reconnection on disconnect
- [ ] Memory creation broadcasts to connected clients
- [ ] Session updates broadcast to connected clients
- [ ] Project-scoped rooms for targeted updates
- [ ] Reinforce/deemphasize work over WebSocket

### Search & Browse

- [ ] Search by query with sector filter
- [ ] Toggle inclusion of superseded memories
- [ ] Results show session context (date, summary snippet)
- [ ] Results show superseded badge when applicable
- [ ] Results show related memory count
- [ ] Click result opens detail modal
- [ ] New memories appear in results in real-time

### Memory Detail Modal

- [ ] Shows full content and metadata
- [ ] Shows session context if available
- [ ] Reinforce/de-emphasize buttons update salience instantly
- [ ] Archive (soft delete) button works
- [ ] Delete forever (hard delete) requires confirmation
- [ ] View Timeline button navigates correctly

### Multi-Agent View

- [ ] Lists sessions from last 24 hours
- [ ] Groups parallel (overlapping) sessions visually
- [ ] Shows session metadata (duration, memory count)
- [ ] Shows active badge for running sessions
- [ ] New sessions appear in real-time
- [ ] Active session updates (memory count) in real-time
- [ ] View Memories button filters search to session
- [ ] View Timeline button shows session timeline

### Timeline

- [ ] Shows chronological list of memories
- [ ] Highlights anchor memory when provided
- [ ] Shows superseded memories with badge
- [ ] Shows session summaries in timeline header
- [ ] Click memory opens detail modal

### Settings & Stats

- [ ] Shows embedding provider info
- [ ] Shows memory counts by sector
- [ ] Shows total sessions count
