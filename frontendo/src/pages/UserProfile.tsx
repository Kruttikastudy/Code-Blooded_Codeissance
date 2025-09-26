import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export const UserProfile = () => {
  return (
    <div className="p-6 space-y-6 max-w-3xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold">Profile</h1>
        <p className="text-muted-foreground">Manage your personal details</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Personal Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid sm:grid-cols-2 gap-4">
            <div>
              <Label>First Name</Label>
              <Input defaultValue="Alex" />
            </div>
            <div>
              <Label>Last Name</Label>
              <Input defaultValue="Johnson" />
            </div>
          </div>
          <div className="grid sm:grid-cols-2 gap-4">
            <div>
              <Label>Email</Label>
              <Input type="email" defaultValue="alex@example.com" />
            </div>
            <div>
              <Label>Phone</Label>
              <Input defaultValue="+91 90000 00000" />
            </div>
          </div>
          <Button className="mt-2">Save Changes</Button>
        </CardContent>
      </Card>
    </div>
  );
};

export default UserProfile;


