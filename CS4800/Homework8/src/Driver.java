public class Driver {
    public static void main(String[] args)
    {
        Snack coke = new Snack("Coke", 12, 8);
        Snack pepsi = new Snack("Pepsi", 12, 8);
        Snack cheetos = new Snack("Cheetos", 12, 8);
        Snack doritos = new Snack("Doritos", 12, 8);
        Snack kitkat = new Snack("KitKat", 12, 8);
        Snack snickers = new Snack("Snickers", 12, 1);

        SnackDispenseHandler head = new SnackDispenseHandler(coke, new SnackDispenseHandler(pepsi,
                new SnackDispenseHandler(cheetos, new SnackDispenseHandler(doritos, new SnackDispenseHandler(kitkat,
                        new SnackDispenseHandler(snickers, new SnackDispenseHandler(null, null)))))));




        VendingMachine machine = new VendingMachine();
        machine.setHeadHandler(head);

        machine.selectSnack("Snickers");
        machine.insertMoney(12);

        machine.selectSnack("Snickers");
        machine.insertMoney(12);
    }
}
