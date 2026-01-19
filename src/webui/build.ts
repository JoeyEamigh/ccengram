import { join } from "node:path";
import postcss from "postcss";
import tailwindcss from "tailwindcss";
import autoprefixer from "autoprefixer";
import { log } from "../utils/log.js";

const WEBUI_DIR = new URL(".", import.meta.url).pathname;

export type BuildOutput = {
  clientJs: string;
  css: string;
};

export async function buildAssets(): Promise<BuildOutput> {
  const start = Date.now();

  const [clientJs, css] = await Promise.all([
    buildClientBundle(),
    buildTailwindCSS(),
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
      "process.env.NODE_ENV": '"production"',
    },
  });

  if (!result.success) {
    log.error("webui", "Failed to build client bundle", {
      logs: result.logs.map((l) => l.message).join("\n"),
    });
    throw new Error(
      "Failed to build client: " + result.logs.map((l) => l.message).join("\n")
    );
  }

  const output = result.outputs[0];
  if (!output) {
    throw new Error("No output from build");
  }
  return await output.text();
}

async function buildTailwindCSS(): Promise<string> {
  const globalsPath = join(WEBUI_DIR, "globals.css");
  const configPath = join(WEBUI_DIR, "tailwind.config.ts");

  const globalsContent = await Bun.file(globalsPath).text();

  const result = await postcss([
    tailwindcss({ config: configPath }),
    autoprefixer(),
  ]).process(globalsContent, {
    from: globalsPath,
    to: "styles.css",
  });

  return result.css;
}
