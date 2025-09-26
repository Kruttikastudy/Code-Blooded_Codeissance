import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Phone, Star, Heart } from "lucide-react";

// Mock data for therapists
const THERAPISTS = [
  {
    id: 1,
    name: "Dr. Sarah Johnson",
    specialty: "Depression & Anxiety",
    rating: 4.9,
    reviews: 124,
    availability: "Available Today",
    image: "https://randomuser.me/api/portraits/women/44.jpg"
  },
  {
    id: 2,
    name: "Dr. Michael Chen",
    specialty: "Trauma & PTSD",
    rating: 4.8,
    reviews: 98,
    availability: "Available Tomorrow",
    image: "https://randomuser.me/api/portraits/men/32.jpg"
  },
  {
    id: 3,
    name: "Dr. Aisha Patel",
    specialty: "Stress Management",
    rating: 4.7,
    reviews: 87,
    availability: "Available Today",
    image: "https://randomuser.me/api/portraits/women/68.jpg"
  },
  {
    id: 4,
    name: "Dr. James Wilson",
    specialty: "Depression & Mood Disorders",
    rating: 4.9,
    reviews: 156,
    availability: "Available Today",
    image: "https://randomuser.me/api/portraits/men/52.jpg"
  },
  {
    id: 5,
    name: "Dr. Elena Rodriguez",
    specialty: "Anxiety & Panic Disorders",
    rating: 4.8,
    reviews: 112,
    availability: "Available Tomorrow",
    image: "https://randomuser.me/api/portraits/women/28.jpg"
  }
];

export const SeekHelp = () => {
  return (
    <div className="p-6 space-y-6 max-w-6xl mx-auto">
      <div className="bg-gradient-to-r from-healing-green/10 to-primary/10 rounded-lg p-6">
        <h1 className="text-2xl font-bold mb-2">Seek Help</h1>
        <p className="text-muted-foreground">Connect with mental health professionals for immediate support</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Available Therapists</CardTitle>
          <CardDescription>
            Select a therapist to schedule a call or immediate consultation
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {THERAPISTS.map((therapist) => (
              <Card key={therapist.id} className="overflow-hidden">
                <div className="aspect-video w-full relative">
                  <img 
                    src={therapist.image} 
                    alt={therapist.name}
                    className="object-cover w-full h-full"
                  />
                  <Badge 
                    className="absolute top-2 right-2"
                    variant={therapist.availability.includes("Today") ? "default" : "secondary"}
                  >
                    {therapist.availability}
                  </Badge>
                </div>
                <CardContent className="p-4">
                  <h3 className="font-semibold text-lg">{therapist.name}</h3>
                  <p className="text-sm text-muted-foreground mb-2">{therapist.specialty}</p>
                  
                  <div className="flex items-center mb-4">
                    <div className="flex items-center">
                      <Star className="h-4 w-4 text-yellow-500 fill-yellow-500" />
                      <span className="ml-1 text-sm font-medium">{therapist.rating}</span>
                    </div>
                    <span className="mx-2 text-muted-foreground">•</span>
                    <span className="text-xs text-muted-foreground">{therapist.reviews} reviews</span>
                  </div>
                  
                  <div className="flex space-x-2">
                    <Button 
                      className="flex-1"
                      onClick={() => {
                        const evt = new CustomEvent("navigate", { detail: { page: "appointments" } });
                        window.dispatchEvent(evt);
                      }}
                    >
                      <Heart className="h-4 w-4 mr-2" />
                      Book Session
                    </Button>
                    <Button 
                      variant="outline" 
                      className="flex-1"
                      onClick={() => window.open("https://meet.google.com/rec-peqo-ehr", "_blank")}
                    >
                      <Phone className="h-4 w-4 mr-2" />
                      Call Now
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SeekHelp;