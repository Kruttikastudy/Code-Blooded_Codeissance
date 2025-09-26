import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowLeft, Stethoscope, Shield, Lock } from "lucide-react";
import { ThemeToggle } from "@/components/ThemeToggle";

interface TherapistLoginProps {
  onBack: () => void;
  onLogin: () => void;
}

export const TherapistLogin = ({ onBack, onLogin }: TherapistLoginProps) => {
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
      onLogin();
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-healing-green/20 to-primary-soft flex items-center justify-center p-4">
      <div className="absolute top-4 left-4">
        <Button variant="ghost" onClick={onBack} className="text-muted-foreground">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>
      </div>
      
      <div className="absolute top-4 right-4">
        <ThemeToggle />
      </div>

      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <div className="p-3 bg-healing-green/10 rounded-full">
              <Stethoscope className="h-8 w-8 text-healing-green" />
            </div>
          </div>
          <h1 className="text-2xl font-bold text-foreground mb-2">Therapist Portal</h1>
          <p className="text-muted-foreground">Professional access to client care tools</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Professional Sign In</CardTitle>
            <CardDescription>
              Secure access to your client dashboard and analytics
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="therapistId">Therapist ID / Email</Label>
                <Input
                  id="therapistId"
                  type="text"
                  placeholder="your.professional@email.com"
                  required
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  placeholder="••••••••"
                  required
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="licenseNumber">License Number</Label>
                <Input
                  id="licenseNumber"
                  type="text"
                  placeholder="Professional License #"
                  required
                />
              </div>

              <Button 
                type="submit" 
                className="w-full bg-healing-green hover:bg-healing-green/90" 
                disabled={isLoading}
              >
                {isLoading ? "Verifying credentials..." : "Professional Sign In"}
              </Button>
              
              <Button type="button" variant="ghost" className="w-full text-sm">
                Contact IT Support
              </Button>
            </form>
          </CardContent>
        </Card>

        <div className="mt-6 p-4 bg-card border rounded-lg">
          <div className="flex items-center space-x-2 text-sm text-muted-foreground mb-2">
            <Shield className="h-4 w-4" />
            <Lock className="h-4 w-4" />
            <span>Professional-grade security & compliance</span>
          </div>
          <p className="text-xs text-muted-foreground">
            This portal is restricted to licensed mental health professionals only. 
            All access is logged and monitored.
          </p>
        </div>
      </div>
    </div>
  );
};