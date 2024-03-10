package CS4800.Polymorphism;

public class Ship {
    private String name;

    private String yearBuilt;

    public Ship(String givenName, String givenYearBuilt) {
        this.name = givenName;
        this.yearBuilt = givenYearBuilt;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getYearBuilt() {
        return yearBuilt;
    }

    public void setYearBuilt(String yearBuilt) {
        this.yearBuilt = yearBuilt;
    }

    public void printShipDetails() {
        System.out.printf("\n\nShip Name: %s", this.name);
        System.out.printf("\nShip Build Date: %s", this.yearBuilt);
    }

    public static void main(String[] args) {

    }
}