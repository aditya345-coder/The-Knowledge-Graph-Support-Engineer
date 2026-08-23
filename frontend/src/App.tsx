import { useState } from "react";
import { useSession } from "./hooks/useSession";
import { Sidebar } from "./components/Sidebar";
import { Chat } from "./pages/Chat";
import { isAuthConfigured } from "./auth/config";
import { AuthGate } from "./auth/AuthGate";
import { useTheme } from "./hooks/useTheme";
import { useWebSearch } from "./hooks/useWebSearch";

function AppContent() {
  const { sessionId, setSessionId, restoreSession } = useSession();
  const [repoUrl, setRepoUrl] = useState<string>();
  const { theme, toggle: toggleTheme } = useTheme();
  const { enabled: webSearchEnabled, toggle: toggleWebSearch } = useWebSearch();

  return (
    <div className="flex h-screen">
      <Sidebar
        sessionId={sessionId}
        onSessionChange={setSessionId}
        onRestoreSession={restoreSession}
        onRepoUrlChange={setRepoUrl}
        theme={theme}
        onThemeToggle={toggleTheme}
      />
      <Chat
        sessionId={sessionId}
        repoUrl={repoUrl}
        webSearchEnabled={webSearchEnabled}
        onWebSearchToggle={toggleWebSearch}
      />
    </div>
  );
}

export default function App() {
  if (isAuthConfigured()) {
    return (
      <AuthGate>
        <AppContent />
      </AuthGate>
    );
  }
  return <AppContent />;
}
