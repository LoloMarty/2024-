import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

public class ProxyServer implements SongService {
    private static ProxyServer proxyServer;
    private static HashMap<String, List<Song>> titleHashmap;
    private static HashMap<String, List<Song>> albumHashmap;
    private static HashMap<String, Song> idHashmap;
    private Song songResult;
    private List<Song> listSongResult;

    private ProxyServer() {
        titleHashmap = new HashMap<String, List<Song>>();
        albumHashmap = new HashMap<String, List<Song>>();
        idHashmap = new HashMap<String, Song>();
    }

    public Song searchById(Integer songID) {
        this.songResult = idHashmap.get(Integer.toString(songID));

        if (songResult == null) {
            System.out.println("Song not cached, fetching from server...");
            Song retrievedSong = Server.getInstance().searchById(songID);
            idHashmap.put(Integer.toString(songID), retrievedSong);
        }

        return idHashmap.get(Integer.toString(songID));
    }

    public List<Song> searchByTitle(String title) {
        this.listSongResult = titleHashmap.get(title);

        if (listSongResult == null) {
            System.out.println("Song not cached, fetching from server...");
            List<Song> retrievedSong = Server.getInstance().searchByTitle(title);
            titleHashmap.put(title, retrievedSong);
        }

        return titleHashmap.get(title);
    }

    public List<Song> searchByAlbum(String album) {
        this.listSongResult = albumHashmap.get(album);

        if (listSongResult == null) {
            System.out.println("Song not cached, fetching from server...");
            List<Song> retrievedSong = Server.getInstance().searchByAlbum(album);
            albumHashmap.put(album, retrievedSong);
        }

        return albumHashmap.get(album);
    }

    public static ProxyServer getInstance() {
        if (proxyServer == null) {
            proxyServer = new ProxyServer();
        }

        return proxyServer;
    }

}
