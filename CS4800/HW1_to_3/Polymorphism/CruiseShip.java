package CS4800.Polymorphism;

public class CruiseShip extends Ship {
    private int maximumPassengers;

    public CruiseShip(String givenName, String givenBuildYear, int givenMaxPassengers) {
        super(givenName, givenBuildYear);
        this.maximumPassengers = givenMaxPassengers;
    }

    public int getMaximumPassengers() {
        return maximumPassengers;
    }

    public void setMaximumPassengers(int maximumPassengers) {
        this.maximumPassengers = maximumPassengers;
    }

    public void printShipDetails() {
        System.out.printf("\n\nShip Name: %s", super.getName());
        System.out.printf("\nShip Max Capacity (Persons): %,d", this.getMaximumPassengers());
    }

    public static void main(String[] args) {

    }
}