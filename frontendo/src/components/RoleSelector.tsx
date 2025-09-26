import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { User, Stethoscope, Shield, Heart } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";

interface RoleSelectorProps {
  onSelectRole: (role: "user" | "therapist") => void;
}

export const RoleSelector = ({ onSelectRole }: RoleSelectorProps) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-soft to-healing-soft flex items-center justify-center p-4">
      <div className="absolute top-4 right-4">
        <ThemeToggle />
      </div>
      
      <div className="w-full max-w-4xl">
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-6">
            <div className="p-4 bg-primary/10 rounded-full">
              <Heart className="h-12 w-12 text-primary" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-foreground mb-4">
            Voice Depression Analysis
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            AI-powered mental health support through voice analysis. 
            Professional, secure, and compassionate care.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 max-w-3xl mx-auto">
          <Card className="hover:shadow-lg transition-all duration-300 cursor-pointer group" 
                onClick={() => onSelectRole("user")}>
            <CardHeader className="text-center pb-4">
              <div className="mx-auto p-4 bg-primary/10 rounded-full w-fit mb-4 group-hover:bg-primary/20 transition-colors">
                <User className="h-8 w-8 text-primary" />
              </div>
              <CardTitle className="text-2xl">I'm a User</CardTitle>
              <CardDescription className="text-base">
                Access personalized mental health insights and support
              </CardDescription>
            </CardHeader>
            <CardContent className="text-center">
              <ul className="text-sm text-muted-foreground space-y-2 mb-6">
                <li>• Voice-based depression screening</li>
                <li>• Personal progress tracking</li>
                <li>• AI-powered insights</li>
                <li>• Therapist consultations</li>
              </ul>
              <Button className="w-full" size="lg">
                Continue as User
              </Button>
            </CardContent>
          </Card>

          <Card className="hover:shadow-lg transition-all duration-300 cursor-pointer group"
                onClick={() => onSelectRole("therapist")}>
            <CardHeader className="text-center pb-4">
              <div className="mx-auto p-4 bg-healing-green/10 rounded-full w-fit mb-4 group-hover:bg-healing-green/20 transition-colors">
                <Stethoscope className="h-8 w-8 text-healing-green" />
              </div>
              <CardTitle className="text-2xl">I'm a Therapist</CardTitle>
              <CardDescription className="text-base">
                Professional tools for client care and analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="text-center">
              <ul className="text-sm text-muted-foreground space-y-2 mb-6">
                <li>• Client progress monitoring</li>
                <li>• Session management</li>
                <li>• Detailed analytics</li>
                <li>• Secure note-taking</li>
              </ul>
              <Button 
                className="w-full bg-healing-green hover:bg-healing-green/90" 
                size="lg"
              >
                Continue as Therapist
              </Button>
            </CardContent>
          </Card>
        </div>

        <div className="text-center mt-12">
          <div className="flex items-center justify-center space-x-2 text-sm text-muted-foreground">
            <Shield className="h-4 w-4" />
            <span>HIPAA Compliant • End-to-End Encrypted • Secure & Private</span>
          </div>
        </div>
      </div>
    </div>
  );
};