import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

public class Server implements SongService {
    private static Server instance;
    private static HashMap<String, List<Song>> titleHashmap;
    private static HashMap<String, List<Song>> albumHashmap;
    private static HashMap<String, Song> idHashmap;

    private final int waitTime = 3000;

    private Server() {
        titleHashmap = new HashMap<String, List<Song>>();
        albumHashmap = new HashMap<String, List<Song>>();
        idHashmap = new HashMap<String, Song>();
    }

    public void addSong(String title, String artist, String album, int id, int Duration) {
        if (titleHashmap == null) {
            titleHashmap = new HashMap<String, List<Song>>();
        }
        if (albumHashmap == null) {
            albumHashmap = new HashMap<String, List<Song>>();
        }
        if (idHashmap == null) {
            idHashmap = new HashMap<String, Song>();
        }

        Song songToAdd = new Song(title, artist, album, id, Duration);

        if (titleHashmap.get(title) == null) {
            titleHashmap.put(title, new LinkedList<Song>());
        }
        titleHashmap.get(title).addFirst(songToAdd);

        if (albumHashmap.get(album) == null) {
            albumHashmap.put(album, new LinkedList<Song>());
        }
        albumHashmap.get(album).addFirst(songToAdd);

        if (idHashmap.get(Integer.toString(id)) == null) {
            idHashmap.put(Integer.toString(id), songToAdd);
        }
    }

    public Song searchById(Integer songID) {
        try {
            Thread.sleep(waitTime);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return idHashmap.get(Integer.toString(songID));
    }

    public List<Song> searchByTitle(String title) {
        try {
            Thread.sleep(waitTime);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return titleHashmap.get(title);
    }

    public List<Song> searchByAlbum(String album) {
        try {
            Thread.sleep(waitTime);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return albumHashmap.get(album);
    }

    public static Server getInstance() {
        if (instance == null) {
            instance = new Server();
        }

        return instance;
    }
}
