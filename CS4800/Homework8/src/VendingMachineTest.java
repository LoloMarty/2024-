import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class VendingMachineTest {
    Snack chips = new Snack("Chips", 12, 2);
    Snack candy = new Snack("Candy", 5, 50);
    private VendingMachine vendingMachine = new VendingMachine();
    SnackDispenseHandler head = new SnackDispenseHandler(chips, new SnackDispenseHandler(candy, new SnackDispenseHandler(null, null)));
    @Test
    void selectSnack() {
        vendingMachine.setHeadHandler(head);
        // Test selecting an existing snack
        vendingMachine.selectSnack("Chips");
        assertEquals("Chips", vendingMachine.getSelectedSnack().getName());
    }

    @Test
    void insertMoney() {
        vendingMachine.setHeadHandler(head);

        vendingMachine.selectSnack("Candy");
        assertEquals(0, vendingMachine.insertMoney(5));

        vendingMachine.selectSnack("Candy");
        assertEquals(45, vendingMachine.insertMoney(50));
    }
}