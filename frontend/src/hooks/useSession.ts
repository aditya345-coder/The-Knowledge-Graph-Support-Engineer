import { useState } from "react";

function generateId(): string {
  return "ui_" + Math.random().toString(36).substring(2, 15);
}

function getInitialSessionId(): string {
  const id = localStorage.getItem("session_id");
  if (id) return id;
  const newId = generateId();
  localStorage.setItem("session_id", newId);
  return newId;
}

export function useSession() {
  const [sessionId, setSessionIdState] = useState<string>(getInitialSessionId);

  const setSessionId = (id: string) => {
    localStorage.setItem("session_id", id);
    setSessionIdState(id);
  };

  const restoreSession = (id: string) => {
    if (id && id !== sessionId) {
      localStorage.setItem("session_id", id);
      setSessionIdState(id);
    }
  };

  return { sessionId, setSessionId, restoreSession };
}
