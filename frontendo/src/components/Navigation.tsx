import { Button } from "@/components/ui/button";
import { ThemeToggle } from "./ThemeToggle";
import { 
  LayoutDashboard, 
  Mic, 
  FileText, 
  User, 
  Calendar,
  Users,
  ClipboardList,
  LogOut,
  Heart
} from "lucide-react";

interface NavigationProps {
  userType: "user" | "therapist";
  currentPage: string;
  onNavigate: (page: string) => void;
  onLogout: () => void;
}

export const Navigation = ({ userType, currentPage, onNavigate, onLogout }: NavigationProps) => {
  const userNavItems = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "quiz", label: "Voice Quiz", icon: Mic },
    { id: "reports", label: "Reports", icon: FileText },
    { id: "appointments", label: "Appointments", icon: Calendar },
    { id: "profile", label: "Profile", icon: User },
  ];

  const therapistNavItems = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "clients", label: "Clients", icon: Users },
    { id: "sessions", label: "Sessions", icon: ClipboardList },
    { id: "appointments", label: "Schedule", icon: Calendar },
    { id: "profile", label: "Profile", icon: User },
  ];

  const navItems = userType === "user" ? userNavItems : therapistNavItems;

  return (
    <nav className="bg-card border-b border-border px-4 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-8">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Heart className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h1 className="text-lg font-semibold">Voice Analysis</h1>
              <p className="text-xs text-muted-foreground capitalize">
                {userType} Portal
              </p>
            </div>
          </div>

          <div className="flex space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <Button
                  key={item.id}
                  variant={currentPage === item.id ? "default" : "ghost"}
                  size="sm"
                  onClick={() => onNavigate(item.id)}
                  className="flex items-center space-x-2"
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{item.label}</span>
                </Button>
              );
            })}
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <ThemeToggle />
          <Button
            variant="ghost"
            size="sm"
            onClick={onLogout}
            className="text-muted-foreground hover:text-destructive"
          >
            <LogOut className="h-4 w-4" />
            <span className="hidden sm:inline ml-2">Logout</span>
          </Button>
        </div>
      </div>
    </nav>
  );
};