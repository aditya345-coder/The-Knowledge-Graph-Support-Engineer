import { useState, useEffect, useMemo } from "react";
import { useAuth0 } from "@auth0/auth0-react";
import { prepareRepo, resumeRepo, freshRepo, cleanupSession, getRepoList, saveRepoListAPI } from "../api/client";
import type { RepoEntry } from "../api/client";
import { useStatus } from "../hooks/useStatus";
import { isAuthConfigured } from "../auth/config";
import { ProgressBar } from "./ProgressBar";

interface Props {
  sessionId: string;
  onSessionChange: (id: string) => void;
  onRestoreSession: (id: string) => void;
  onRepoUrlChange: (url: string) => void;
  theme: "light" | "dark";
  onThemeToggle: () => void;
}

function AuthSection() {
  const { loginWithRedirect, logout, isAuthenticated, user } = useAuth0();
  const [authError, setAuthError] = useState<string | null>(null);

  if (!isAuthConfigured()) return null;

  if (!isAuthenticated) {
    return (
      <div>
        <button
          onClick={async () => {
            try {
              setAuthError(null);
              await loginWithRedirect();
            } catch (err) {
              setAuthError(
                err instanceof Error ? err.message : "Failed to sign in. Check Auth0 configuration."
              );
            }
          }}
          className="w-full py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-900 text-sm cursor-pointer dark:bg-gray-200 dark:text-gray-900 dark:hover:bg-gray-300"
        >
          Sign In
        </button>
        {authError && (
          <p className="text-red-600 text-xs mt-1">{authError}</p>
        )}
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 pb-2 border-b dark:border-gray-700">
      {user?.picture && (
        <img
          src={user.picture}
          alt=""
          className="w-8 h-8 rounded-full"
        />
      )}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate dark:text-gray-100">{user?.name || user?.email || "User"}</p>
      </div>
      <button
        onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })}
        className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 cursor-pointer"
      >
        Log out
      </button>
    </div>
  );
}

function loadRepoList(): RepoEntry[] {
  try {
    return JSON.parse(localStorage.getItem("repo_list") || "[]");
  } catch {
    return [];
  }
}

function saveRepoList(list: RepoEntry[]) {
  localStorage.setItem("repo_list", JSON.stringify(list));
}

function repoNameFromUrl(url: string): string {
  return url.replace(/^https?:\/\/github\.com\//, "").replace(/\.git$/, "") || url;
}

function generateId(): string {
  return "ui_" + Math.random().toString(36).substring(2, 15);
}

export function Sidebar({ sessionId, onSessionChange, onRestoreSession, onRepoUrlChange, theme, onThemeToggle }: Props) {
  const [repoUrl, setRepoUrl] = useState("");
  const [showResumeDialog, setShowResumeDialog] = useState(false);
  const [interruptedSession, setInterruptedSession] = useState<{ completed_phases: string[]; percent?: number } | null>(null);
  const [repoList, setRepoList] = useState<RepoEntry[]>(loadRepoList);

  const status = useStatus(sessionId);
  const stage = status?.data?.stage;
  const isPreparing = stage === "running";

  const statusRepoUrl = status?.data?.repo_url ?? "";
  const effectiveRepoUrl = useMemo(
    () => repoUrl || statusRepoUrl,
    [repoUrl, statusRepoUrl],
  );

  // Sync migration and resume detection from status
  useEffect(() => {
    if (!status?.data) return;
    if (status.data.repo_url) {
      onRepoUrlChange(status.data.repo_url);
      const currentList = loadRepoList();
      if (!currentList.some((r) => r.sessionId === sessionId)) {
        const name = repoNameFromUrl(status.data.repo_url);
        const updated = [...currentList, {
          sessionId,
          repoUrl: status.data.repo_url,
          name,
          preparedAt: new Date().toISOString(),
        }];
        saveRepoList(updated);
        // Use queueMicrotask to avoid synchronous setState in effect
        queueMicrotask(() => setRepoList(updated));
      }
    }
    if (status.data.stage === "running") {
      const data = status.data;
      queueMicrotask(() => {
        setInterruptedSession({
          completed_phases: data.completed_phases || [],
          percent: data.percent,
        });
        setShowResumeDialog(true);
      });
    }
  }, [status?.data, sessionId, onRepoUrlChange]);

  // Sync repo list from backend on mount
  useEffect(() => {
    getRepoList().then((backendRepos) => {
      if (backendRepos.length === 0) return;
      const localList = loadRepoList();
      const localIds = new Set(localList.map((r) => r.sessionId));
      const newRepos = backendRepos.filter((r) => !localIds.has(r.sessionId));
      const merged = [...localList, ...newRepos];
      setRepoList(merged);
      saveRepoList(merged);

      // If current sessionId doesn't match any saved repo,
      // auto-select the most recent one from the server
      if (merged.length > 0 && !merged.some((r) => r.sessionId === sessionId)) {
        const sorted = [...merged].sort(
          (a, b) => new Date(b.preparedAt).getTime() - new Date(a.preparedAt).getTime(),
        );
        onRestoreSession(sorted[0].sessionId);
      }
    }).catch(() => {});
  }, []);

  const handlePrepare = async () => {
    if (!repoUrl.trim()) return;
    try {
      const result = await prepareRepo(repoUrl.trim(), sessionId);
      if (result.status === "interrupted") {
        setInterruptedSession({
          completed_phases: result.completed_phases || [],
        });
        setShowResumeDialog(true);
        return;
      }
      const name = repoNameFromUrl(repoUrl.trim());
      const updated = repoList.filter((r) => r.sessionId !== sessionId);
      updated.push({ sessionId, repoUrl: repoUrl.trim(), name, preparedAt: new Date().toISOString() });
      setRepoList(updated);
      saveRepoList(updated);
      saveRepoListAPI(updated).catch(() => {});
    } catch {
      // API call failed — status will remain undefined, isPreparing stays true briefly
      // but useStatus will clear it on next poll
    }
  };

  const handleResume = async () => {
    setShowResumeDialog(false);
    try {
      await resumeRepo(sessionId);
    } catch {
      // status-based isPreparing handles UI state
    }
  };

  const handleFresh = async () => {
    setShowResumeDialog(false);
    try {
      await freshRepo(repoUrl.trim(), sessionId);
    } catch {
      // status-based isPreparing handles UI state
    }
  };

  const handleSelectRepo = (entry: RepoEntry) => {
    setRepoUrl(entry.repoUrl);
    onRepoUrlChange(entry.repoUrl);
    onSessionChange(entry.sessionId);
  };

  const handleDeleteRepo = async (entry: RepoEntry) => {
    try {
      await cleanupSession(entry.sessionId);
    } catch {
      // best-effort cleanup
    }
    const updated = repoList.filter((r) => r.sessionId !== entry.sessionId);
    setRepoList(updated);
    saveRepoList(updated);
    saveRepoListAPI(updated).catch(() => {});
    if (entry.sessionId === sessionId) {
      if (updated.length > 0) {
        handleSelectRepo(updated[0]);
      } else {
        setRepoUrl("");
        onRepoUrlChange("");
        onSessionChange(generateId());
      }
    }
  };

  const handleAddNew = () => {
    setRepoUrl("");
    onRepoUrlChange("");
    onSessionChange(generateId());
  };

  const handleRepoUrlInput = (url: string) => {
    setRepoUrl(url);
    onRepoUrlChange(url);
  };

  const isComplete = stage === "complete";
  const isError = stage === "error";

  return (
    <aside className="w-72 bg-gray-50 border-r p-4 flex flex-col gap-4 dark:bg-gray-900 dark:border-gray-700">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold text-lg dark:text-gray-100">Repository</h2>
        <button
          onClick={onThemeToggle}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          className="p-1.5 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 cursor-pointer"
        >
          {theme === "dark" ? (
            <svg className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
            </svg>
          ) : (
            <svg className="w-5 h-5 text-gray-600" fill="currentColor" viewBox="0 0 24 24">
              <path d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
            </svg>
          )}
        </button>
      </div>

      <AuthSection />

      {repoList.length > 0 && (
        <div className="space-y-1">
          <label className="text-sm text-gray-600 dark:text-gray-400">Saved Repos</label>
          <div className="max-h-40 overflow-y-auto space-y-1">
            {repoList.map((entry) => (
              <div
                key={entry.sessionId}
                className={`flex items-center gap-1 px-2 py-1.5 rounded-lg text-sm cursor-pointer ${
                  entry.sessionId === sessionId
                    ? "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
                    : "hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300"
                }`}
              >
                <button
                  onClick={() => handleSelectRepo(entry)}
                  className="flex-1 text-left truncate"
                  title={entry.repoUrl}
                >
                  {entry.name}
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); handleDeleteRepo(entry); }}
                  title="Delete repo"
                  className="p-0.5 hover:text-red-600 dark:hover:text-red-400 flex-shrink-0 cursor-pointer"
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-2">
        <label className="text-sm text-gray-600 dark:text-gray-400">Repo URL</label>
        <input
          type="text"
          value={effectiveRepoUrl}
          onChange={(e) => handleRepoUrlInput(e.target.value)}
          placeholder="owner/repo"
          disabled={isPreparing}
          className="w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 dark:bg-gray-800 dark:border-gray-600 dark:text-gray-100 dark:placeholder-gray-400"
        />
      </div>

      <div className="flex gap-2">
        <button
          onClick={handlePrepare}
          disabled={isPreparing || !repoUrl.trim()}
          className="flex-1 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 cursor-pointer"
        >
          {isPreparing ? "Preparing..." : "Prepare"}
        </button>
        <button
          onClick={handleAddNew}
          disabled={isPreparing}
          className="px-3 py-2 border border-gray-300 rounded-lg text-sm hover:bg-gray-100 disabled:opacity-50 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-700 cursor-pointer"
          title="Add new repo"
        >
          + New
        </button>
      </div>

      {showResumeDialog && interruptedSession && (
        <div className="p-3 border rounded-lg bg-yellow-50 text-sm dark:bg-yellow-900/30 dark:border-yellow-700">
          <p className="font-medium text-yellow-800 dark:text-yellow-200">
            Previous session was interrupted
          </p>
          {interruptedSession.completed_phases.length > 0 && (
            <p className="text-yellow-700 text-xs mt-1 dark:text-yellow-300">
              Completed: {interruptedSession.completed_phases.join(", ")}
            </p>
          )}
          <div className="mt-3 flex gap-2">
            <button
              onClick={handleResume}
              className="px-3 py-1 bg-blue-600 text-white rounded text-sm cursor-pointer"
            >
              Resume
            </button>
            <button
              onClick={handleFresh}
              className="px-3 py-1 bg-gray-200 text-gray-700 rounded text-sm dark:bg-gray-700 dark:text-gray-200 cursor-pointer"
            >
              Start Fresh
            </button>
          </div>
        </div>
      )}

      {isPreparing && status?.data && (
        <ProgressBar
          percent={status.data.percent || 0}
          message={status.data.message || stage || "Processing..."}
          etaSeconds={status.data.eta_seconds}
        />
      )}

      <div className="mt-2">
        <span
          className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
            isComplete
              ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
              : isError
              ? "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
              : isPreparing
              ? "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
              : "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"
          }`}
        >
          {isComplete ? "Ready" : isError ? "Error" : isPreparing ? "Processing" : "Not started"}
        </span>
      </div>

      {isError && status?.data?.message && (
        <div className="w-full space-y-1">
          <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
            <div className="bg-red-600 h-2.5 rounded-full" style={{ width: "100%" }} />
          </div>
          <div className="text-xs text-red-600 dark:text-red-400">
            {status.data.message}
          </div>
          <button
            onClick={handleAddNew}
            className="text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 cursor-pointer"
          >
            Try a different repo
          </button>
        </div>
      )}
    </aside>
  );
}
