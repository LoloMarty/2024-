package AbstractFactory;

public class DriverProgram {

    public static void main(String[] args) {
        Customer john = new Customer("John", "No Restriction");
        john.getMeal();

        Customer tran = new Customer("Tran", "Paleo");
        tran.getMeal();

        Customer pablo = new Customer("Pablo", "Vegan");
        pablo.getMeal();

        Customer alex = new Customer("Alex", "Nut Allergy");
        alex.getMeal();

        Customer patrick = new Customer("Patrick", "Vegan");
        patrick.getMeal();

        Customer legolas = new Customer("Legolas", "Paleo");
        legolas.getMeal();
    }
}
