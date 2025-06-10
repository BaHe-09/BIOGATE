import { Ionicons } from '@expo/vector-icons';
import { usePathname, useRouter } from 'expo-router';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useTheme } from '../context/ThemeContext'; // Ajusta la ruta según tu estructura

export default function BottomNavBar() {
  const router = useRouter();
  const pathname = usePathname();
  const { temaOscuro } = useTheme();

  const isActive = (path: string) => pathname === path;

  const colors = {
    light: {
      background: '#ffffff',
      inactiveIcon: '#aaaaaa',
      inactiveText: '#666666',
      active: '#007AFF',
      shadow: '#000000',
      shadowOpacity: 0.08,
    },
    dark: {
      background: '#1e1e1e',
      inactiveIcon: '#757575',
      inactiveText: '#9e9e9e',
      active: '#0a84ff',
      shadow: '#000000',
      shadowOpacity: 0.3,
    }
  };

  const currentColors = temaOscuro ? colors.dark : colors.light;

  return (
    <View style={styles.wrapper}>
      <View style={[
        styles.container,
        {
          backgroundColor: currentColors.background,
          shadowColor: currentColors.shadow,
          shadowOpacity: currentColors.shadowOpacity,
        }
      ]}>
        <TouchableOpacity onPress={() => router.replace('/home')} style={styles.item}>
          <Ionicons
            name="mail"
            size={24}
            color={isActive('/home') ? currentColors.active : currentColors.inactiveIcon}
          />
          <Text style={[
            styles.textBase,
            {
              color: isActive('/home') ? currentColors.active : currentColors.inactiveText,
              fontWeight: isActive('/home') ? '600' : '500'
            }
          ]}>Historial</Text>
        </TouchableOpacity>

        <TouchableOpacity onPress={() => router.replace('/usuarios')} style={styles.item}>
          <Ionicons
            name="person"
            size={24}
            color={isActive('/usuarios') ? currentColors.active : currentColors.inactiveIcon}
          />
          <Text style={[
            styles.textBase,
            {
              color: isActive('/usuarios') ? currentColors.active : currentColors.inactiveText,
              fontWeight: isActive('/usuarios') ? '600' : '500'
            }
          ]}>Usuarios</Text>
        </TouchableOpacity>

        <TouchableOpacity onPress={() => router.replace('/camera')} style={styles.item}>
          <Ionicons
            name="camera"
            size={24}
            color={isActive('/camera') ? currentColors.active : currentColors.inactiveIcon}
          />
          <Text style={[
            styles.textBase,
            {
              color: isActive('/camera') ? currentColors.active : currentColors.inactiveText,
              fontWeight: isActive('/camera') ? '600' : '500'
            }
          ]}>Cámara</Text>
        </TouchableOpacity>

        <TouchableOpacity onPress={() => router.replace('/administracion')} style={styles.item}>
          <Ionicons
            name="bar-chart"
            size={24}
            color={isActive('/administracion') ? currentColors.active : currentColors.inactiveIcon}
          />
          <Text style={[
            styles.textBase,
            {
              color: isActive('/administracion') ? currentColors.active : currentColors.inactiveText,
              fontWeight: isActive('/administracion') ? '600' : '500'
            }
          ]}>Admin</Text>
        </TouchableOpacity>

         <TouchableOpacity onPress={() => router.replace('/config')} style={styles.item}>
          <Ionicons
            name="settings"
            size={24}
            color={isActive('/config') ? currentColors.active : currentColors.inactiveIcon}
          />
          <Text style={[
            styles.textBase,
            {
              color: isActive('/config') ? currentColors.active : currentColors.inactiveText,
              fontWeight: isActive('/config') ? '600' : '500'
            }
          ]}>Ajustes</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    position: 'absolute',
    bottom: 30,
    left: 20,
    right: 20,
    alignItems: 'center',
    zIndex: 10,
  },
  container: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    borderRadius: 30,
    paddingHorizontal: 20,
    paddingVertical: 10,
    width: '100%',
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 6,
    elevation: 5,
  },
  item: {
    alignItems: 'center',
    flex: 1,
  },
  textBase: {
    fontSize: 12,
    marginTop: 3,
  },
});
