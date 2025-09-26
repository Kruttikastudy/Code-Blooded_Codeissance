import { useState } from "react";
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
import { Navigation } from "@/components/Navigation";

const queryClient = new QueryClient();

type AppState = "roleSelection" | "userLogin" | "therapistLogin" | "userApp" | "therapistApp";
type UserType = "user" | "therapist";
type CurrentPage = "dashboard" | "quiz" | "reports" | "appointments" | "profile" | "clients" | "sessions";

const App = () => {
  const [appState, setAppState] = useState<AppState>("roleSelection");
  const [userType, setUserType] = useState<UserType>("user");
  const [currentPage, setCurrentPage] = useState<CurrentPage>("dashboard");

  const handleRoleSelection = (role: UserType) => {
    setUserType(role);
    setAppState(role === "user" ? "userLogin" : "therapistLogin");
  };

  const handleLogin = () => {
    setAppState(userType === "user" ? "userApp" : "therapistApp");
    setCurrentPage("dashboard");
  };

  const handleLogout = () => {
    setAppState("roleSelection");
    setCurrentPage("dashboard");
  };

  const handleNavigation = (page: string) => {
    setCurrentPage(page as CurrentPage);
  };

  const renderCurrentPage = () => {
    if (appState === "userApp") {
      switch (currentPage) {
        case "dashboard":
          return <UserDashboard />;
        case "quiz":
          return <VoiceQuiz />;
        case "reports":
          return <div className="p-6 text-center text-muted-foreground">Reports page coming soon...</div>;
        case "appointments":
          return <div className="p-6 text-center text-muted-foreground">Appointments page coming soon...</div>;
        case "profile":
          return <div className="p-6 text-center text-muted-foreground">Profile page coming soon...</div>;
        default:
          return <UserDashboard />;
      }
    } else if (appState === "therapistApp") {
      switch (currentPage) {
        case "dashboard":
          return <TherapistDashboard />;
        case "clients":
          return <div className="p-6 text-center text-muted-foreground">Client list page coming soon...</div>;
        case "sessions":
          return <div className="p-6 text-center text-muted-foreground">Session notes page coming soon...</div>;
        case "appointments":
          return <div className="p-6 text-center text-muted-foreground">Schedule page coming soon...</div>;
        case "profile":
          return <div className="p-6 text-center text-muted-foreground">Profile page coming soon...</div>;
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
