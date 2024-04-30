import org.junit.Test;

import java.util.LinkedList;
import java.util.List;

import static org.junit.Assert.*;

public class ServerTest {

    @Test
    public void addSong() {
        Server server = Server.getInstance();
        server.addSong("Title1", "Artist1", "Album1", 1, 180);

        Song searchedSong = server.searchById(1);
        assertNotNull(searchedSong);
        assertEquals("Title1", searchedSong.getTitle());
        assertEquals("Artist1", searchedSong.getArtist());
        assertEquals("Album1", searchedSong.getAlbum());
        assertEquals(180, searchedSong.getDuration());

        List<Song> songsByTitle = server.searchByTitle("Title1");
        assertNotNull(songsByTitle);
        assertEquals(1, songsByTitle.size());
        assertEquals("Title1", songsByTitle.get(0).getTitle());

        List<Song> songsByAlbum = server.searchByAlbum("Album1");
        assertNotNull(songsByAlbum);
        assertEquals(1, songsByAlbum.size());
        assertEquals("Album1", songsByAlbum.get(0).getAlbum());
    }

    @Test
    public void searchById() {
        Server server = Server.getInstance();

        Song testSong = new Song("Test Song", "Test Artist", "Test Album", -1, 120);
        server.addSong("Test Song", "Test Artist", "Test Album", -1, 120);

        Song result = server.searchById(-1);

        assertEquals(testSong.getTitle(), result.getTitle());
    }

    @Test
    public void searchByTitle() {
        Server server = Server.getInstance();

        List<Song> testSong = new LinkedList<Song>();
        testSong.addFirst(new Song("Test Song", "Test Artist", "Test Album", 1, 120));
        server.addSong("Test Song", "Test Artist", "Test Album", 1, 120);

        List<Song> result = server.searchByTitle("Test Song");

        assertEquals(testSong.getFirst().getTitle(), result.getFirst().getTitle());
    }

    @Test
    public void searchByAlbum() {
        Server server = Server.getInstance();

        List<Song> testSong = new LinkedList<Song>();
        testSong.addFirst(new Song("Test Song", "Test Artist", "Test Album", 1, 120));
        server.addSong("Test Song", "Test Artist", "Test Album", 1, 120);

        List<Song> result = server.searchByAlbum("Test Album");

        assertEquals(testSong.getFirst().getTitle(), result.getFirst().getTitle());
    }

    @Test
    public void getInstance() {
        Server instance1 = Server.getInstance();
        Server instance2 = Server.getInstance();

        assertSame(instance1, instance2);

        assertNotNull(instance1);
        assertNotNull(instance2);
    }
}