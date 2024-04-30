import java.sql.Driver;
import java.util.List;

public class DriverProgram {
    public static void printSongInfo(Song song) {
        System.out.println();
        System.out.printf("\nName: %s", song.getTitle());
        System.out.printf("\nAlbum: %s", song.getAlbum());
        System.out.printf("\nArtist: %s", song.getArtist());
        System.out.printf("\nSongID: %d", song.getId());
        System.out.printf("\nDuration (s): %d\n\n", song.getDuration());
    }

    public static void printSongListInfo(List<Song> listOfSongs) {
        for (Song song : listOfSongs) {
            printSongInfo(song);
        }
    }

    public static void main(String[] args) {
        Server server = Server.getInstance();
        ProxyServer proxyServer = ProxyServer.getInstance();

        server.addSong("Paint It Blue", "Charley Crockett", "Cowboy Jams", 1, 120);
        server.addSong("Separate Ways", "Journey", "80s Hits", 2, 240);
        server.addSong("Life is a Highway", "Pixar", "Cars Movie OST", 3, 120);
        server.addSong("21", "Sam Hunt", "Sad Country", 4, 120);
        server.addSong("Paint It Black", "The Animals", "Vietnam War Music", 5, 240);
        server.addSong("Free Bird", "Lynrd Skynyrd", "Patriotic Jams", 6, 600);

        List<Song> returnedSongs = proxyServer.searchByTitle("Paint It Blue");
        // Slow Retrieval
        DriverProgram.printSongListInfo(returnedSongs);

        returnedSongs = proxyServer.searchByTitle("Paint It Blue");
        // Fast Retrieval (Cached)
        DriverProgram.printSongListInfo(returnedSongs);

        Song returnedSong = proxyServer.searchById(6);
        // Fast Retrieval (Cached)
        DriverProgram.printSongInfo(returnedSong);

    }
}
