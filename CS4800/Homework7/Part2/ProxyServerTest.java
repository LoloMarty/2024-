import static org.junit.Assert.*;
import java.util.*;
public class ProxyServerTest {

    @org.junit.Test
    public void searchById() {
        ProxyServer proxyServer = ProxyServer.getInstance();
        Server server = Server.getInstance();

        Song testSong = new Song("Test Song", "Test Artist", "Test Album", 1, 120);
        server.addSong("Test Song", "Test Artist", "Test Album", 1, 120);

        Song result = proxyServer.searchById(1);

        assertEquals(testSong.getTitle(), result.getTitle());
    }

    @org.junit.Test
    public void searchByTitle() {
        ProxyServer proxyServer = ProxyServer.getInstance();
        Server server = Server.getInstance();


        List<Song> testSong = new LinkedList<Song>();
        testSong.addFirst(new Song("Test Song", "Test Artist", "Test Album", 1, 120));
        server.addSong("Test Song", "Test Artist", "Test Album", 1, 120);

        List<Song> result = proxyServer.searchByTitle("Test Song");

        assertEquals(testSong.getFirst().getTitle(), result.getFirst().getTitle());
    }

    @org.junit.Test
    public void searchByAlbum() {
        ProxyServer proxyServer = ProxyServer.getInstance();
        Server server = Server.getInstance();

        List<Song> testSong = new LinkedList<Song>();
        testSong.addFirst(new Song("Test Song", "Test Artist", "Test Album", 1, 120));
        server.addSong("Test Song", "Test Artist", "Test Album", 1, 120);

        List<Song> result = proxyServer.searchByAlbum("Test Album");

        assertEquals(testSong.getFirst().getTitle(), result.getFirst().getTitle());
    }

    @org.junit.Test
    public void getInstance() {
        ProxyServer instance1 = ProxyServer.getInstance();
        ProxyServer instance2 = ProxyServer.getInstance();

        assertSame(instance1, instance2);

        assertNotNull(instance1);
        assertNotNull(instance2);
    }
}