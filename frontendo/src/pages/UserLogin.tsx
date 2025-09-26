import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { ArrowLeft, Heart, Shield, Lock, Phone, User, Mail } from "lucide-react";
import { ThemeToggle } from "@/components/ThemeToggle";
import { useToast } from "@/hooks/use-toast";

interface SpeedDialContact {
  name: string;
  phone: string;
  email: string;
}

interface UserLoginProps {
  onBack: () => void;
  onLogin: (isNewUser?: boolean) => void;
}

export const UserLogin = ({ onBack, onLogin }: UserLoginProps) => {
  const [isLoading, setIsLoading] = useState(false);
  const [consentGiven, setConsentGiven] = useState(false);
  const [speedDialContacts, setSpeedDialContacts] = useState<SpeedDialContact[]>([
    { name: "", phone: "", email: "" },
    { name: "", phone: "", email: "" }
  ]);
  const { toast } = useToast();

  const updateSpeedDialContact = (index: number, field: keyof SpeedDialContact, value: string) => {
    const updated = [...speedDialContacts];
    updated[index] = { ...updated[index], [field]: value };
    setSpeedDialContacts(updated);
  };

  const validateSpeedDialContacts = () => {
    return speedDialContacts.every(contact => 
      contact.name.trim() && contact.phone.trim() && contact.email.trim()
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!consentGiven) {
      toast({
        variant: "destructive",
        title: "Consent Required",
        description: "Please agree to the terms and consent for audio recording.",
      });
      return;
    }
    
    setIsLoading(true);
    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
      onLogin();
    }, 1500);
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!consentGiven) {
      toast({
        variant: "destructive",
        title: "Consent Required",
        description: "Please agree to the terms and consent for audio recording.",
      });
      return;
    }

    if (!validateSpeedDialContacts()) {
      toast({
        variant: "destructive",
        title: "Speed Dial Contacts Required",
        description: "Please fill in all speed dial contact details (name, phone, email).",
      });
      return;
    }
    
    setIsLoading(true);
    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
      try {
        // Persist speed dial contacts locally for now
        localStorage.setItem("speedDialContacts", JSON.stringify(speedDialContacts));
      } catch {}
      onLogin(true); // Pass true to indicate new user
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-soft to-healing-soft flex items-center justify-center p-4">
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
            <div className="p-3 bg-primary/10 rounded-full">
              <Heart className="h-8 w-8 text-primary" />
            </div>
          </div>
          <h1 className="text-2xl font-bold text-foreground mb-2">Welcome Back</h1>
          <p className="text-muted-foreground">Your mental health journey continues here</p>
        </div>

        <Tabs defaultValue="login" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="login">Login</TabsTrigger>
            <TabsTrigger value="register">Register</TabsTrigger>
          </TabsList>
          
          <TabsContent value="login">
            <Card>
              <CardHeader>
                <CardTitle>Sign In</CardTitle>
                <CardDescription>
                  Enter your credentials to access your dashboard
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="email">Email</Label>
                    <Input
                      id="email"
                      type="email"
                      placeholder="your@email.com"
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
                  
                  <div className="flex items-center space-x-2">
                    <Checkbox 
                      id="consent" 
                      checked={consentGiven}
                      onCheckedChange={(checked) => setConsentGiven(checked === true)}
                    />
                    <Label htmlFor="consent" className="text-sm">
                      I consent to audio recording and data processing for mental health analysis
                    </Label>
                  </div>

                  <Button type="submit" className="w-full" disabled={isLoading}>
                    {isLoading ? "Signing in..." : "Sign In"}
                  </Button>
                  
                  <Button type="button" variant="ghost" className="w-full text-sm">
                    Forgot your password?
                  </Button>
                </form>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="register">
            <Card>
              <CardHeader>
                <CardTitle>Create Account</CardTitle>
                <CardDescription>
                  Join our secure mental health platform
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleRegister} className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="firstName">First Name</Label>
                      <Input
                        id="firstName"
                        placeholder="John"
                        required
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="lastName">Last Name</Label>
                      <Input
                        id="lastName"
                        placeholder="Doe"
                        required
                      />
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="registerEmail">Email</Label>
                    <Input
                      id="registerEmail"
                      type="email"
                      placeholder="your@email.com"
                      required
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="registerPassword">Password</Label>
                    <Input
                      id="registerPassword"
                      type="password"
                      placeholder="••••••••"
                      required
                    />
                  </div>

                  {/* Speed Dial Contacts */}
                  <div className="space-y-4">
                    <div className="flex items-center space-x-2">
                      <Phone className="h-4 w-4 text-primary" />
                      <Label className="text-sm font-medium">Speed Dial Contacts (Required)</Label>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Add 2 emergency contacts who can be reached quickly in case of crisis
                    </p>
                    
                    {speedDialContacts.map((contact, index) => (
                      <div key={index} className="p-4 border rounded-lg space-y-3">
                        <div className="flex items-center space-x-2">
                          <User className="h-4 w-4 text-muted-foreground" />
                          <Label className="text-sm font-medium">Contact {index + 1}</Label>
                        </div>
                        
                        <div className="space-y-2">
                          <Label htmlFor={`contactName${index}`}>Name</Label>
                          <Input
                            id={`contactName${index}`}
                            placeholder="Contact Name"
                            value={contact.name}
                            onChange={(e) => updateSpeedDialContact(index, 'name', e.target.value)}
                            required
                          />
                        </div>
                        
                        <div className="space-y-2">
                          <Label htmlFor={`contactPhone${index}`}>Phone Number</Label>
                          <Input
                            id={`contactPhone${index}`}
                            type="tel"
                            placeholder="+1 (555) 123-4567"
                            value={contact.phone}
                            onChange={(e) => updateSpeedDialContact(index, 'phone', e.target.value)}
                            required
                          />
                        </div>
                        
                        <div className="space-y-2">
                          <Label htmlFor={`contactEmail${index}`}>Email</Label>
                          <Input
                            id={`contactEmail${index}`}
                            type="email"
                            placeholder="contact@email.com"
                            value={contact.email}
                            onChange={(e) => updateSpeedDialContact(index, 'email', e.target.value)}
                            required
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <Checkbox 
                      id="registerConsent" 
                      checked={consentGiven}
                      onCheckedChange={(checked) => setConsentGiven(checked === true)}
                    />
                    <Label htmlFor="registerConsent" className="text-sm">
                      I agree to the terms and consent to audio recording for mental health analysis
                    </Label>
                  </div>

                  <Button type="submit" className="w-full" disabled={isLoading}>
                    {isLoading ? "Creating account..." : "Create Account"}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        <div className="mt-6 p-4 bg-card border rounded-lg">
          <div className="flex items-center space-x-2 text-sm text-muted-foreground">
            <Shield className="h-4 w-4" />
            <Lock className="h-4 w-4" />
            <span>Your data is encrypted and HIPAA compliant</span>
          </div>
        </div>
      </div>
    </div>
  );
};