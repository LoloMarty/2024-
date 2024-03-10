package CS4800.Polymorphism;

public class DriverProgram {

    public static void main(String[] args) {
        Ship normalShip = new Ship("Maria Ave", "2002");
        CruiseShip cruiseShip = new CruiseShip("Balaegyr", "1985", 3000);
        CargoShip cargoShip = new CargoShip("Coastal Packer", "2018", 600);

        Ship[] fleet = { normalShip, cruiseShip, cargoShip };

        for (int ship = 0; ship < fleet.length; ship++) {
            fleet[ship].printShipDetails();
        }
    }
}
