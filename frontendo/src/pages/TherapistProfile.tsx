import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export const TherapistProfile = () => {
  return (
    <div className="p-6 space-y-6 max-w-3xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold">Therapist Profile</h1>
        <p className="text-muted-foreground">Update your professional details</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Professional Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid sm:grid-cols-2 gap-4">
            <div>
              <Label>Full Name</Label>
              <Input defaultValue="Dr. Aisha Sharma" />
            </div>
            <div>
              <Label>Specialization</Label>
              <Input defaultValue="Clinical Psychologist" />
            </div>
          </div>
          <div className="grid sm:grid-cols-2 gap-4">
            <div>
              <Label>Email</Label>
              <Input type="email" defaultValue="aisha@example.com" />
            </div>
            <div>
              <Label>Phone</Label>
              <Input defaultValue="+91 98888 88888" />
            </div>
          </div>
          <Button className="mt-2">Save Changes</Button>
        </CardContent>
      </Card>
    </div>
  );
};

export default TherapistProfile;


