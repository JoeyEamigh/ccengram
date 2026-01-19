import { hydrateRoot } from "react-dom/client";
import { App } from "./components/App.js";

declare global {
  interface Window {
    __INITIAL_DATA__: unknown;
  }
}

const initialData = window.__INITIAL_DATA__;
const rootElement = document.getElementById("root");

if (rootElement) {
  hydrateRoot(
    rootElement,
    <App url={window.location.pathname} initialData={initialData} />
  );
}
