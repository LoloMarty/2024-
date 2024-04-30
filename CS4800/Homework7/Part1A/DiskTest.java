import org.junit.Test;

import static org.junit.Assert.*;

public class DiskTest {

    @Test
    public void getDocument() {
        CharString expected = Disk.getDocument();
        CharString actual = Disk.getDocument();

        assertNotNull(actual);
        assertEquals(expected, actual);
    }
}