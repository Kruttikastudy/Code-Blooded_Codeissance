import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { 
  Users, 
  TrendingUp, 
  TrendingDown, 
  Calendar,
  Search,
  FileText,
  Phone,
  AlertTriangle,
  CheckCircle,
  Clock,
  Activity,
  Filter
} from "lucide-react";

export const TherapistDashboard = () => {
  // Mock data - in real app, this would come from API
  const clients = [
    {
      id: 1,
      name: "Alex Johnson",
      lastScore: 35,
      trend: "improving",
      lastAnalysis: "2024-01-15",
      riskLevel: "Low",
      status: "stable"
    },
    {
      id: 2,
      name: "Sarah Chen",
      lastScore: 65,
      trend: "concerning",
      lastAnalysis: "2024-01-14",
      riskLevel: "Moderate",
      status: "needs-attention"
    },
    {
      id: 3,
      name: "Michael Davis",
      lastScore: 28,
      trend: "stable",
      lastAnalysis: "2024-01-13",
      riskLevel: "Low",
      status: "stable"
    },
    {
      id: 4,
      name: "Emma Wilson",
      lastScore: 78,
      trend: "declining",
      lastAnalysis: "2024-01-12",
      riskLevel: "High",
      status: "urgent"
    },
  ];

  const todayAppointments = [
    { time: "09:00 AM", client: "Alex Johnson", type: "Follow-up" },
    { time: "11:30 AM", client: "Sarah Chen", type: "Check-in" },
    { time: "02:00 PM", client: "New Client", type: "Initial Assessment" },
    { time: "04:30 PM", client: "Emma Wilson", type: "Crisis Support" },
  ];

  const getRiskBadgeProps = (riskLevel: string) => {
    switch (riskLevel) {
      case "Low":
        return { variant: "default" as const, icon: CheckCircle };
      case "Moderate":
        return { variant: "secondary" as const, icon: Clock };
      case "High":
        return { variant: "destructive" as const, icon: AlertTriangle };
      default:
        return { variant: "outline" as const, icon: Clock };
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "improving":
        return <TrendingDown className="h-4 w-4 text-success" />;
      case "declining":
      case "concerning":
        return <TrendingUp className="h-4 w-4 text-destructive" />;
      default:
        return <Activity className="h-4 w-4 text-muted-foreground" />;
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="bg-gradient-to-r from-healing-green/10 to-primary/10 rounded-lg p-6">
        <h1 className="text-2xl font-bold mb-2">Therapist Dashboard</h1>
        <p className="text-muted-foreground">
          Overview of your clients' mental health analytics and upcoming sessions
        </p>
      </div>

      {/* Statistics Cards */}
      <div className="grid md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Users className="h-5 w-5 text-primary" />
              <span className="text-sm font-medium">Total Clients</span>
            </div>
            <div className="mt-2">
              <div className="text-2xl font-bold">24</div>
              <p className="text-xs text-muted-foreground">Active this month</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5 text-warning" />
              <span className="text-sm font-medium">High Risk</span>
            </div>
            <div className="mt-2">
              <div className="text-2xl font-bold text-destructive">3</div>
              <p className="text-xs text-muted-foreground">Require attention</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <TrendingDown className="h-5 w-5 text-success" />
              <span className="text-sm font-medium">Improving</span>
            </div>
            <div className="mt-2">
              <div className="text-2xl font-bold text-success">12</div>
              <p className="text-xs text-muted-foreground">Positive trends</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Calendar className="h-5 w-5 text-primary" />
              <span className="text-sm font-medium">Today</span>
            </div>
            <div className="mt-2">
              <div className="text-2xl font-bold">{todayAppointments.length}</div>
              <p className="text-xs text-muted-foreground">Appointments</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Client List */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Client Overview</CardTitle>
                  <CardDescription>
                    Recent analyses and risk assessments
                  </CardDescription>
                </div>
                <div className="flex space-x-2">
                  <Button variant="outline" size="sm">
                    <Filter className="h-4 w-4 mr-2" />
                    Filter
                  </Button>
                </div>
              </div>
              <div className="flex space-x-2 mt-4">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search clients..."
                    className="pl-9"
                  />
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {clients.map((client) => {
                  const badgeProps = getRiskBadgeProps(client.riskLevel);
                  const BadgeIcon = badgeProps.icon;
                  
                  return (
                    <div 
                      key={client.id} 
                      className="p-4 rounded-lg border bg-card/50 hover:bg-card transition-colors cursor-pointer"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                          <div>
                            <h4 className="font-medium">{client.name}</h4>
                            <p className="text-sm text-muted-foreground">
                              Last analysis: {new Date(client.lastAnalysis).toLocaleDateString()}
                            </p>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-3">
                          <div className="text-right">
                            <div className="text-lg font-semibold">
                              {client.lastScore}
                            </div>
                            <div className="flex items-center space-x-1">
                              {getTrendIcon(client.trend)}
                              <span className="text-xs text-muted-foreground capitalize">
                                {client.trend}
                              </span>
                            </div>
                          </div>
                          
                          <Badge variant={badgeProps.variant}>
                            <BadgeIcon className="h-3 w-3 mr-1" />
                            {client.riskLevel}
                          </Badge>
                        </div>
                      </div>
                      
                      <div className="flex justify-end mt-3 space-x-2">
                        <Button variant="ghost" size="sm">
                          <FileText className="h-4 w-4 mr-1" />
                          Notes
                        </Button>
                        <Button variant="ghost" size="sm">
                          <Phone className="h-4 w-4 mr-1" />
                          Call
                        </Button>
                        <Button variant="outline" size="sm">
                          View Details
                        </Button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Today's Schedule */}
        <div>
          <Card className="h-fit">
            <CardHeader>
              <CardTitle>Today's Schedule</CardTitle>
              <CardDescription>
                Upcoming appointments and sessions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {todayAppointments.map((appointment, index) => (
                  <div key={index} className="flex items-center space-x-3 p-3 rounded-lg bg-muted/30">
                    <div className="text-sm font-medium text-primary min-w-[70px]">
                      {appointment.time}
                    </div>
                    <div className="flex-1">
                      <p className="font-medium text-sm">{appointment.client}</p>
                      <p className="text-xs text-muted-foreground">{appointment.type}</p>
                    </div>
                    <Button variant="ghost" size="sm">
                      Join
                    </Button>
                  </div>
                ))}
              </div>
              
              <Button className="w-full mt-4" variant="outline">
                <Calendar className="h-4 w-4 mr-2" />
                View Full Calendar
              </Button>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button className="w-full justify-start" variant="ghost">
                <FileText className="h-4 w-4 mr-2" />
                Create Session Note
              </Button>
              <Button className="w-full justify-start" variant="ghost">
                <Users className="h-4 w-4 mr-2" />
                Add New Client
              </Button>
              <Button className="w-full justify-start" variant="ghost">
                <Activity className="h-4 w-4 mr-2" />
                Generate Report
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};