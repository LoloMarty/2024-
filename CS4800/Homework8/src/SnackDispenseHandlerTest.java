import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SnackDispenseHandlerTest {

    @Test
    public void testHandleRequest_SuccessfulDispense() {
        // Arrange
        Snack givenSnack = new Snack("Chocolate Bar", 0, 10);
        SnackDispenseHandler handler = new SnackDispenseHandler(givenSnack, null);

        // Act
        Snack dispensedSnack = handler.handleRequest("Chocolate Bar");

        // Assert
        assertEquals(9, givenSnack.getQuantity());
        assertEquals("Chocolate Bar", dispensedSnack.getName());
    }
}