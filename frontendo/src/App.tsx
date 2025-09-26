import { useEffect, useState } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { RoleSelector } from "@/components/RoleSelector";
import { UserLogin } from "@/pages/UserLogin";
import { TherapistLogin } from "@/pages/TherapistLogin";
import { UserDashboard } from "@/pages/UserDashboard";
import { TherapistDashboard } from "@/pages/TherapistDashboard";
import { VoiceQuiz } from "@/pages/VoiceQuiz";
import Reports from "@/pages/Reports";
import Appointments from "@/pages/Appointments";
import UserProfile from "@/pages/UserProfile";
import { Navigation } from "@/components/Navigation";
import TherapistClients from "@/pages/TherapistClients";
import TherapistSessions from "@/pages/TherapistSessions";
import TherapistSchedule from "@/pages/TherapistSchedule";
import TherapistProfile from "@/pages/TherapistProfile";

const queryClient = new QueryClient();

type AppState = "roleSelection" | "userLogin" | "therapistLogin" | "userApp" | "therapistApp";
type UserType = "user" | "therapist";
type CurrentPage = "dashboard" | "quiz" | "reports" | "appointments" | "profile" | "clients" | "sessions";

const App = () => {
  const [appState, setAppState] = useState<AppState>("roleSelection");
  const [userType, setUserType] = useState<UserType>("user");
  const [currentPage, setCurrentPage] = useState<CurrentPage>("dashboard");
  const [isNewUser, setIsNewUser] = useState(false);

  // Listen to simple navigation events from child components (e.g., dashboard quick actions)
  // so they can request page changes without prop drilling.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail as { page?: string } | undefined;
      if (detail?.page) {
        setCurrentPage(detail.page as CurrentPage);
      }
    };
    window.addEventListener("navigate", handler as EventListener);
    const firstQuizHandler = () => {
      setIsNewUser(false);
    };
    window.addEventListener("first-quiz-complete", firstQuizHandler as EventListener);
    return () => {
      window.removeEventListener("navigate", handler as EventListener);
      window.removeEventListener("first-quiz-complete", firstQuizHandler as EventListener);
    };
  }, []);

  const handleRoleSelection = (role: UserType) => {
    setUserType(role);
    setAppState(role === "user" ? "userLogin" : "therapistLogin");
  };

  const handleLogin = (newUser?: boolean) => {
    setAppState(userType === "user" ? "userApp" : "therapistApp");
    setIsNewUser(newUser || false);
    setCurrentPage(newUser ? "quiz" : "dashboard");
  };

  const handleLogout = () => {
    setAppState("roleSelection");
    setCurrentPage("dashboard");
    setIsNewUser(false);
  };

  const handleNavigation = (page: string) => {
    setCurrentPage(page as CurrentPage);
  };

  const renderCurrentPage = () => {
    if (appState === "userApp") {
      // For new users, show welcome screen on dashboard
      if (currentPage === "dashboard" && isNewUser) {
        return <UserDashboard isNewUser={true} />;
      }
      
      switch (currentPage) {
        case "dashboard":
          return <UserDashboard isNewUser={false} />;
        case "quiz":
          return <VoiceQuiz />;
        case "reports":
          return <Reports />;
        case "appointments":
          return <Appointments />;
        case "profile":
          return <UserProfile />;
        default:
          return <UserDashboard isNewUser={false} />;
      }
    } else if (appState === "therapistApp") {
      switch (currentPage) {
        case "dashboard":
          return <TherapistDashboard />;
        case "clients":
          return <TherapistClients />;
        case "sessions":
          return <TherapistSessions />;
        case "appointments":
          return <TherapistSchedule />;
        case "profile":
          return <TherapistProfile />;
        default:
          return <TherapistDashboard />;
      }
    }
    return null;
  };

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          
          {appState === "roleSelection" && (
            <RoleSelector onSelectRole={handleRoleSelection} />
          )}
          
          {appState === "userLogin" && (
            <UserLogin 
              onBack={() => setAppState("roleSelection")} 
              onLogin={handleLogin}
            />
          )}
          
          {appState === "therapistLogin" && (
            <TherapistLogin 
              onBack={() => setAppState("roleSelection")} 
              onLogin={handleLogin}
            />
          )}
          
          {(appState === "userApp" || appState === "therapistApp") && (
            <div className="min-h-screen bg-background">
              <Navigation 
                userType={userType}
                currentPage={currentPage}
                onNavigate={handleNavigation}
                onLogout={handleLogout}
              />
              {renderCurrentPage()}
            </div>
          )}
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

export default App;
