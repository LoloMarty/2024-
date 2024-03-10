package CS4800.Polymorphism;

public class CargoShip extends Ship {
    private int cargoCapacity_InTons;

    public CargoShip(String givenName, String givenYearBuilt, int givenCargoCapacity_InTons) {
        super(givenName, givenYearBuilt);
        this.cargoCapacity_InTons = givenCargoCapacity_InTons;
    }

    public int getCargoCapacity_InTons() {
        return cargoCapacity_InTons;
    }

    public void setCargoCapacity_InTons(int cargoCapacity_InTons) {
        this.cargoCapacity_InTons = cargoCapacity_InTons;
    }

    public void printShipDetails() {
        System.out.printf("\n\nShip Name: %s", super.getName());
        System.out.printf("\nShip Capacity (Tons): %,d", this.getCargoCapacity_InTons());
    }

}
