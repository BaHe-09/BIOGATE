import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import {
  Dimensions,
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { BarChart, PieChart } from 'react-native-chart-kit';
import { useTheme } from '../context/ThemeContext';

const screenWidth = Dimensions.get('window').width - 48;

export default function EstadisticasScreen() {
  const router = useRouter();
  const { temaOscuro } = useTheme();
  const dynamicStyles = temaOscuro ? darkStyles : lightStyles;

  const [modo, setModo] = useState<'none' | 'horarios' | 'areas' | 'extras'>('none');
  const [seleccion, setSeleccion] = useState<string | null>(null);
  const [tipoGrafica, setTipoGrafica] = useState<'barras' | 'pastel'>('barras');

  // Datos ficticios para horarios
  const datosMatutino = {
    labels: ['Lun', 'Mar', 'Mié', 'Jue', 'Vie'],
    datasets: [
      { 
        data: [120, 145, 128, 160, 140],
        colors: [
          (opacity = 1) => `rgba(76, 175, 80, ${opacity})`,
          (opacity = 1) => `rgba(76, 175, 80, ${opacity})`,
          (opacity = 1) => `rgba(76, 175, 80, ${opacity})`,
          (opacity = 1) => `rgba(76, 175, 80, ${opacity})`,
          (opacity = 1) => `rgba(76, 175, 80, ${opacity})`
        ]
      }
    ],
  };

  const datosVespertino = {
    labels: ['Lun', 'Mar', 'Mié', 'Jue', 'Vie'],
    datasets: [
      { 
        data: [95, 105, 88, 120, 110],
        colors: [
          (opacity = 1) => `rgba(33, 150, 243, ${opacity})`,
          (opacity = 1) => `rgba(33, 150, 243, ${opacity})`,
          (opacity = 1) => `rgba(33, 150, 243, ${opacity})`,
          (opacity = 1) => `rgba(33, 150, 243, ${opacity})`,
          (opacity = 1) => `rgba(33, 150, 243, ${opacity})`
        ]
      }
    ],
  };

  // ===== DATOS MEJORADOS PARA ENTRADAS/SALIDAS =====
  const datosEntradas = {
    labels: ['Lun', 'Mar', 'Mié', 'Jue', 'Vie'],
    datasets: [
      {
        data: [85, 92, 88, 95, 90], // % de puntualidad
        colors: [
          (opacity = 1) => `rgba(76, 175, 80, ${opacity})`,
          (opacity = 1) => `rgba(76, 175, 80, ${opacity})`,
          (opacity = 1) => `rgba(76, 175, 80, ${opacity})`,
          (opacity = 1) => `rgba(76, 175, 80, ${opacity})`,
          (opacity = 1) => `rgba(76, 175, 80, ${opacity})`
        ]
      }
    ]
  };

  const datosSalidas = {
    labels: ['Lun', 'Mar', 'Mié', 'Jue', 'Vie'],
    datasets: [
      {
        data: [92, 95, 90, 97, 94], // % de salidas a tiempo
        colors: [
          (opacity = 1) => `rgba(33, 150, 243, ${opacity})`,
          (opacity = 1) => `rgba(33, 150, 243, ${opacity})`,
          (opacity = 1) => `rgba(33, 150, 243, ${opacity})`,
          (opacity = 1) => `rgba(33, 150, 243, ${opacity})`,
          (opacity = 1) => `rgba(33, 150, 243, ${opacity})`
        ]
      }
    ]
  };

  const pastelEntrada = [
    { 
      name: 'Puntuales (8:00-8:10)', 
      population: 78, 
      color: '#4CAF50', 
      legendFontColor: temaOscuro ? '#fff' : '#000', 
      legendFontSize: 12 
    },
    { 
      name: 'Tardanzas leves (8:11-8:30)', 
      population: 15, 
      color: '#FFC107', 
      legendFontColor: temaOscuro ? '#fff' : '#000', 
      legendFontSize: 12 
    },
    { 
      name: 'Tardanzas graves (+30 min)', 
      population: 5, 
      color: '#F44336', 
      legendFontColor: temaOscuro ? '#fff' : '#000', 
      legendFontSize: 12 
    },
    { 
      name: 'Sin registro', 
      population: 2, 
      color: '#9E9E9E', 
      legendFontColor: temaOscuro ? '#fff' : '#000', 
      legendFontSize: 12 
    },
  ];

  const pastelSalida = [
    { 
      name: 'A tiempo (17:00-17:10)', 
      population: 85, 
      color: '#2196F3', 
      legendFontColor: temaOscuro ? '#fff' : '#000', 
      legendFontSize: 12 
    },
    { 
      name: 'Salidas tempranas (16:30-16:59)', 
      population: 10, 
      color: '#FF9800', 
      legendFontColor: temaOscuro ? '#fff' : '#000', 
      legendFontSize: 12 
    },
    { 
      name: 'Salidas muy tempranas (-30 min)', 
      population: 3, 
      color: '#F44336', 
      legendFontColor: temaOscuro ? '#fff' : '#000', 
      legendFontSize: 12 
    },
    { 
      name: 'Sin registro', 
      population: 2, 
      color: '#9E9E9E', 
      legendFontColor: temaOscuro ? '#fff' : '#000', 
      legendFontSize: 12 
    },
  ];

  // ===== DATOS MEJORADOS PARA HORAS EXTRAS =====
  const datosHorasExtras = {
    labels: ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun'],
    datasets: [
      {
        data: [45, 38, 52, 47, 60, 55],
        colors: [
          (opacity = 1) => `rgba(255, 152, 0, ${opacity})`,
          (opacity = 1) => `rgba(255, 152, 0, ${opacity})`,
          (opacity = 1) => `rgba(255, 152, 0, ${opacity})`,
          (opacity = 1) => `rgba(255, 152, 0, ${opacity})`,
          (opacity = 1) => `rgba(255, 152, 0, ${opacity})`,
          (opacity = 1) => `rgba(255, 152, 0, ${opacity})`
        ]
      }
    ]
  };

  const pastelHorasExtras = [
    { 
      name: 'Horas normales', 
      population: 420, 
      color: '#4CAF50', 
      legendFontColor: temaOscuro ? '#fff' : '#000', 
      legendFontSize: 12 
    },
    { 
      name: 'Horas extras diurnas', 
      population: 65, 
      color: '#FFC107', 
      legendFontColor: temaOscuro ? '#fff' : '#000', 
      legendFontSize: 12 
    },
    { 
      name: 'Horas extras nocturnas', 
      population: 25, 
      color: '#FF9800', 
      legendFontColor: temaOscuro ? '#fff' : '#000', 
      legendFontSize: 12 
    },
    { 
      name: 'Horas festivas', 
      population: 10, 
      color: '#F44336', 
      legendFontColor: temaOscuro ? '#fff' : '#000', 
      legendFontSize: 12 
    },
    { 
      name: 'Horas dominicales', 
      population: 5, 
      color: '#9C27B0', 
      legendFontColor: temaOscuro ? '#fff' : '#000', 
      legendFontSize: 12 
    },
  ];

  const chartConfig = {
    backgroundColor: temaOscuro ? '#1e1e1e' : '#f5faff',
    backgroundGradientFrom: temaOscuro ? '#1e1e1e' : '#f5faff',
    backgroundGradientTo: temaOscuro ? '#1e1e1e' : '#f5faff',
    color: () => temaOscuro ? '#ffffff' : '#000000',
    labelColor: () => temaOscuro ? '#ffffff' : '#000000',
    strokeWidth: 2,
    decimalPlaces: 0,
    barPercentage: 0.5,
    propsForDots: {
      r: "6",
      strokeWidth: "2",
      stroke: "#ffa726"
    }
  };

  const renderGrafica = () => {
    if (!seleccion) return null;

    if (modo === 'horarios') {
      const datos = seleccion === 'Matutino' ? datosMatutino : datosVespertino;
      if (tipoGrafica === 'barras') {
        return (
          <BarChart
            data={datos}
            width={screenWidth}
            height={220}
            chartConfig={chartConfig}
            fromZero
            style={{ borderRadius: 12, marginTop: 20 }}
            yAxisLabel=""
            yAxisSuffix=" pers"
            verticalLabelRotation={30}
          />
        );
      } else {
        return (
          <PieChart
            data={[
              { name: 'Asistencia', population: 85, color: '#4CAF50', legendFontColor: temaOscuro ? '#fff' : '#000', legendFontSize: 12 },
              { name: 'Faltas', population: 8, color: '#F44336', legendFontColor: temaOscuro ? '#fff' : '#000', legendFontSize: 12 },
              { name: 'Vacaciones', population: 5, color: '#2196F3', legendFontColor: temaOscuro ? '#fff' : '#000', legendFontSize: 12 },
              { name: 'Incapacidades', population: 2, color: '#FFC107', legendFontColor: temaOscuro ? '#fff' : '#000', legendFontSize: 12 },
            ]}
            width={screenWidth}
            height={220}
            chartConfig={chartConfig}
            accessor="population"
            backgroundColor="transparent"
            paddingLeft="15"
            style={{ borderRadius: 12, marginTop: 20 }}
          />
        );
      }
    }

    if (modo === 'areas') {
      if (tipoGrafica === 'barras') {
        const datos = seleccion === 'Entrada' ? datosEntradas : datosSalidas;
        return (
          <BarChart
            data={datos}
            width={screenWidth}
            height={220}
            chartConfig={chartConfig}
            fromZero
            style={{ borderRadius: 12, marginTop: 20 }}
            yAxisLabel=""
            yAxisSuffix="%"
            verticalLabelRotation={30}
          />
        );
      } else {
        const datos = seleccion === 'Entrada' ? pastelEntrada : pastelSalida;
        return (
          <PieChart
            data={datos}
            width={screenWidth}
            height={220}
            chartConfig={chartConfig}
            accessor="population"
            backgroundColor="transparent"
            paddingLeft="15"
            style={{ borderRadius: 12, marginTop: 20 }}
          />
        );
      }
    }

    if (modo === 'extras') {
      if (tipoGrafica === 'barras') {
        return (
          <BarChart
            data={datosHorasExtras}
            width={screenWidth}
            height={220}
            chartConfig={chartConfig}
            fromZero
            style={{ borderRadius: 12, marginTop: 20 }}
            yAxisLabel=""
            yAxisSuffix=" hrs"
            verticalLabelRotation={30}
          />
        );
      } else {
        return (
          <PieChart
            data={pastelHorasExtras}
            width={screenWidth}
            height={220}
            chartConfig={chartConfig}
            accessor="population"
            backgroundColor="transparent"
            paddingLeft="15"
            style={{ borderRadius: 12, marginTop: 20 }}
          />
        );
      }
    }

    return null;
  };

  return (
    <SafeAreaView style={[{ flex: 1 }, dynamicStyles.container]}>
      <TouchableOpacity
        onPress={() => router.replace('/administracion')}
        style={[styles.backButton, temaOscuro ? styles.backDark : styles.backLight]}
      >
        <Ionicons name="arrow-back" size={24} color={temaOscuro ? '#fff' : '#000'} />
      </TouchableOpacity>

      <View style={styles.body}>
        <Text style={[styles.title, dynamicStyles.text]}>Estadísticas</Text>

        {modo === 'none' && (
          <>
            <TouchableOpacity style={[styles.bigButton, styles.blue]} onPress={() => setModo('horarios')}>
              <Ionicons name="time-outline" size={26} color="#fff" style={{ marginRight: 10 }} />
              <Text style={styles.buttonText}>Horarios</Text>
            </TouchableOpacity>

            <TouchableOpacity style={[styles.bigButton, styles.green]} onPress={() => setModo('areas')}>
              <Ionicons name="locate-outline" size={26} color="#fff" style={{ marginRight: 10 }} />
              <Text style={styles.buttonText}>Entradas/Salidas</Text>
            </TouchableOpacity>

            <TouchableOpacity style={[styles.bigButton, styles.orange]} onPress={() => setModo('extras')}>
              <Ionicons name="alert-circle-outline" size={26} color="#fff" style={{ marginRight: 10 }} />
              <Text style={styles.buttonText}>Horas Extras</Text>
            </TouchableOpacity>
          </>
        )}

        {modo === 'horarios' && (
          <View style={styles.selectionArea}>
            <Text style={[styles.subtitle, dynamicStyles.text]}>🕐 Selecciona un horario:</Text>
            <View style={styles.row}>
              <TouchableOpacity 
                onPress={() => setSeleccion('Matutino')} 
                style={[styles.selector, seleccion === 'Matutino' && styles.selected]}
              >
                <Ionicons name="sunny-outline" size={20} color="#fff" style={{ marginRight: 6 }} />
                <Text style={styles.selectorText}>Matutino</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                onPress={() => setSeleccion('Vespertino')} 
                style={[styles.selector, seleccion === 'Vespertino' && styles.selected]}
              >
                <Ionicons name="moon-outline" size={20} color="#fff" style={{ marginRight: 6 }} />
                <Text style={styles.selectorText}>Vespertino</Text>
              </TouchableOpacity>
            </View>
            <Text style={[styles.subtitle, dynamicStyles.text]}>📈 Tipo de gráfica:</Text>
            <View style={styles.row}>
              <TouchableOpacity 
                onPress={() => setTipoGrafica('barras')} 
                style={[styles.selector, tipoGrafica === 'barras' && styles.selected]}
              >
                <Ionicons name="bar-chart-outline" size={20} color="#fff" style={{ marginRight: 6 }} />
                <Text style={styles.selectorText}>Barras</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                onPress={() => setTipoGrafica('pastel')} 
                style={[styles.selector, tipoGrafica === 'pastel' && styles.selected]}
              >
                <Ionicons name="pie-chart-outline" size={20} color="#fff" style={{ marginRight: 6 }} />
                <Text style={styles.selectorText}>Pastel</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {modo === 'areas' && (
          <View style={styles.selectionArea}>
            <Text style={[styles.subtitle, dynamicStyles.text]}>🏢 Selecciona un área:</Text>
            <View style={styles.row}>
              <TouchableOpacity 
                onPress={() => setSeleccion('Entrada')} 
                style={[styles.selector, seleccion === 'Entrada' && styles.selected]}
              >
                <Ionicons name="log-in-outline" size={20} color="#fff" style={{ marginRight: 6 }} />
                <Text style={styles.selectorText}>Entrada</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                onPress={() => setSeleccion('Salida')} 
                style={[styles.selector, seleccion === 'Salida' && styles.selected]}
              >
                <Ionicons name="log-out-outline" size={20} color="#fff" style={{ marginRight: 6 }} />
                <Text style={styles.selectorText}>Salida</Text>
              </TouchableOpacity>
            </View>
            <Text style={[styles.subtitle, dynamicStyles.text]}>📊 Tipo de gráfica:</Text>
            <View style={styles.row}>
              <TouchableOpacity 
                onPress={() => setTipoGrafica('barras')} 
                style={[styles.selector, tipoGrafica === 'barras' && styles.selected]}
              >
                <Ionicons name="bar-chart-outline" size={20} color="#fff" style={{ marginRight: 6 }} />
                <Text style={styles.selectorText}>Barras</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                onPress={() => setTipoGrafica('pastel')} 
                style={[styles.selector, tipoGrafica === 'pastel' && styles.selected]}
              >
                <Ionicons name="pie-chart-outline" size={20} color="#fff" style={{ marginRight: 6 }} />
                <Text style={styles.selectorText}>Pastel</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {modo === 'extras' && (
          <View style={styles.selectionArea}>
            <Text style={[styles.subtitle, dynamicStyles.text]}>⏱️ Horas extras por:</Text>
            <View style={styles.row}>
              <TouchableOpacity 
                onPress={() => {
                  setSeleccion('Mensual');
                  setTipoGrafica('barras');
                }} 
                style={[styles.selector, seleccion === 'Mensual' && styles.selected]}
              >
                <Ionicons name="calendar-outline" size={20} color="#fff" style={{ marginRight: 6 }} />
                <Text style={styles.selectorText}>Mensual</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                onPress={() => {
                  setSeleccion('Tipo');
                  setTipoGrafica('pastel');
                }} 
                style={[styles.selector, seleccion === 'Tipo' && styles.selected]}
              >
                <Ionicons name="pricetags-outline" size={20} color="#fff" style={{ marginRight: 6 }} />
                <Text style={styles.selectorText}>Por tipo</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {renderGrafica()}

        {modo !== 'none' && (
          <TouchableOpacity 
            style={[styles.bigButton, styles.red, { marginTop: 20 }]} 
            onPress={() => {
              setModo('none');
              setSeleccion(null);
            }}
          >
            <Ionicons name="arrow-undo-outline" size={26} color="#fff" style={{ marginRight: 10 }} />
            <Text style={styles.buttonText}>Volver</Text>
          </TouchableOpacity>
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  body: {
    marginTop: 100,
    paddingHorizontal: 24,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: '900',
    marginBottom: 30,
  },
  subtitle: {
    fontSize: 17,
    fontWeight: '600',
    marginTop: 15,
    marginBottom: 6,
    textAlign: 'center',
  },
  backButton: {
    position: 'absolute',
    top: 40,
    left: 20,
    padding: 10,
    borderRadius: 50,
    zIndex: 10,
  },
  backLight: {
    backgroundColor: 'rgba(0,0,0,0.05)',
  },
  backDark: {
    backgroundColor: 'rgba(255,255,255,0.1)',
  },
  bigButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 18,
    paddingHorizontal: 20,
    borderRadius: 20,
    marginBottom: 20,
    width: '100%',
    justifyContent: 'center',
    elevation: 4,
  },
  buttonText: {
    fontSize: 18,
    fontWeight: '800',
    color: '#fff',
  },
  blue: {
    backgroundColor: '#007AFF',
  },
  green: {
    backgroundColor: '#34C759',
  },
  orange: {
    backgroundColor: '#FF9500',
  },
  red: {
    backgroundColor: '#FF3B30',
  },
  selectionArea: {
    marginTop: 10,
    width: '100%',
    alignItems: 'center',
  },
  row: {
    flexDirection: 'row',
    gap: 10,
    marginVertical: 10,
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  selector: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 16,
    backgroundColor: '#444',
    borderRadius: 16,
    margin: 4,
  },
  selected: {
    backgroundColor: '#007AFF',
  },
  selectorText: {
    color: '#fff',
    fontWeight: '600',
  },
});

const lightStyles = StyleSheet.create({
  container: {
    backgroundColor: '#f5faff',
  },
  text: {
    color: '#111',
  },
});

const darkStyles = StyleSheet.create({
  container: {
    backgroundColor: '#1C1C1E',
  },
  text: {
    color: '#fff',
  },
});