import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class StateOfVendingMachineTest {

    @Test
    public void testInitialState() {
        StateOfVendingMachine vendingMachine = new StateOfVendingMachine();
        assertEquals("Idle", vendingMachine.getState());
    }

    @Test
    public void testSetStateIdle() {
        StateOfVendingMachine vendingMachine = new StateOfVendingMachine();
        vendingMachine.setState_Idle();
        assertEquals("Idle", vendingMachine.getState());
    }

    @Test
    public void testSetStateWaitForMoney() {
        StateOfVendingMachine vendingMachine = new StateOfVendingMachine();
        vendingMachine.setState_WaitForMoney();
        assertEquals("Waiting For Money", vendingMachine.getState());
    }

    @Test
    public void testSetStateDispensingSnack() {
        StateOfVendingMachine vendingMachine = new StateOfVendingMachine();
        vendingMachine.setState_DispensingSnack();
        assertEquals("Dispensing Snack", vendingMachine.getState());
    }

    @Test
    public void testDispenseSnack() {
        StateOfVendingMachine vendingMachine = new StateOfVendingMachine();
        vendingMachine.dispenseSnack();
    }
}
