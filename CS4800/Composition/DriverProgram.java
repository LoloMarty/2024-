package CS4800.Composition;

public class DriverProgram {
    public static void main(String[] args) {

        Folder app_Folder_Subfolders = new Folder();
        app_Folder_Subfolders.setName("app");

        app_Folder_Subfolders.addFolder("config");
        app_Folder_Subfolders.addFolder("controllers");
        app_Folder_Subfolders.addFolder("library");
        app_Folder_Subfolders.addFolder("migrations");
        app_Folder_Subfolders.addFolder("models");
        app_Folder_Subfolders.addFolder("views");

        Folder sourceFiles_Folder = new Folder();
        sourceFiles_Folder.setName("Source Files");
        sourceFiles_Folder.addFile(".htaccess");
        sourceFiles_Folder.addFile(".htrouter.php");
        sourceFiles_Folder.addFile("index.html");

        sourceFiles_Folder.addFolder(".phalcon");
        sourceFiles_Folder.addFolder(app_Folder_Subfolders);
        sourceFiles_Folder.addFolder("cache");
        sourceFiles_Folder.addFolder("public");

        Folder demo1_Folder = new Folder();
        demo1_Folder.setName("demo1");
        demo1_Folder.addFolder(sourceFiles_Folder);
        demo1_Folder.addFolder("Include Path");
        demo1_Folder.addFolder("Remote Files");

        System.out.println();

        demo1_Folder.printEntireDirectory();

        System.out.println();
    }

}
