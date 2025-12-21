from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(
        doc.pagesize[0] - doc.rightMargin,
        0.65 * inch,
        f"Página {doc.page}",
    )
    canvas.restoreState()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Genera un PDF A4 con respuestas detalladas (con referencias al código) "
            "para preguntas de defensa del proyecto."
        )
    )
    parser.add_argument(
        "--out",
        type=str,
        default="docs/respuestas_detalladas_proyecto_rl.pdf",
        help="Ruta de salida del PDF.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        spaceAfter=10,
    )

    h_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=6,
    )

    sh_style = ParagraphStyle(
        "SubHeading",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        spaceBefore=8,
        spaceAfter=4,
    )

    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        spaceAfter=6,
    )

    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=14,
        bulletIndent=6,
        spaceAfter=2,
    )

    code = ParagraphStyle(
        "Code",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=9.5,
        leading=11,
        spaceBefore=6,
        spaceAfter=10,
    )

    margin = 1.0 * inch  # márgenes normales ~ 2.54 cm

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title="Respuestas detalladas – Proyecto RL CarRacing (DQN)",
        author="Grupo 4",
    )

    story: list = []

    story.append(Paragraph("Respuestas detalladas para defensa del proyecto", title_style))
    story.append(
        Paragraph(
            f"Proyecto: DQN aplicado a CarRacing-v3 (Gymnasium). Fecha: {date.today().isoformat()}. Formato: A4.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Este documento responde en detalle 3 preguntas frecuentes, citando explícitamente archivos y funciones del repositorio y mostrando fragmentos de código clave.",
            body,
        )
    )

    # ---------------------------------------------------------------------
    # 1) Procesamiento
    # ---------------------------------------------------------------------
    story.append(Paragraph("1) ¿Qué tipo de procesamiento se está utilizando?", h_style))

    story.append(Paragraph("Resumen", sh_style))
    story.append(
        Paragraph(
            "El proyecto aplica un preprocesamiento visual clásico para RL con entradas de imagen: "
            "(1) conversión a escala de grises, (2) resize a 96×96, (3) normalización a [0,1], y "
            "(4) apilado temporal de 4 frames (frame stacking). Esto reduce dimensionalidad, elimina color y "
            "permite capturar movimiento (información temporal) sin recurrir a una RNN.",
            body,
        )
    )

    story.append(Paragraph("¿Dónde está implementado en el código?", sh_style))
    story.append(Paragraph("• Preprocesamiento del frame: src/preprocesamiento.py → process_state_image(frame)", bullet, bulletText="•"))
    story.append(Paragraph("• Stack inicial: src/preprocesamiento.py → generar_stack_inicial(frame)", bullet, bulletText="•"))
    story.append(Paragraph("• Actualización del stack: src/preprocesamiento.py → actualizar_stack(stack, frame)", bullet, bulletText="•"))
    story.append(Paragraph("• Construcción del estado final: src/preprocesamiento.py → generar_state_frame_stack_from_queue(queue)", bullet, bulletText="•"))
    story.append(
        Paragraph(
            "Estas funciones se usan en el loop de entrenamiento (src/entrenamiento.py) y también en evaluación (tests/probar_modelo.py).",
            body,
        )
    )

    story.append(Paragraph("Orden de procesamiento (paso a paso)", sh_style))
    story.append(Paragraph("1. El entorno entrega un frame (obs) RGB de 96×96×3.", bullet, bulletText="•"))
    story.append(Paragraph("2. Se transforma a escala de grises y se hace resize a 96×96.", bullet, bulletText="•"))
    story.append(Paragraph("3. Se convierte a float32 y se normaliza dividiendo entre 255.", bullet, bulletText="•"))
    story.append(Paragraph("4. Se mantiene un deque de longitud 4 con los últimos frames procesados.", bullet, bulletText="•"))
    story.append(Paragraph("5. El estado final es np.stack(queue, axis=0) con forma (4, 96, 96).", bullet, bulletText="•"))

    story.append(Paragraph("Fundamento (por qué se hace así)", sh_style))
    story.append(
        Paragraph(
            "Este pipeline es habitual en DQN visual: la escala de grises reduce canales y complejidad, "
            "la normalización mejora estabilidad numérica, y el stacking permite inferir velocidad/dinámica. "
            "En el repositorio se cita como referencia el estilo Atari-DQN (Mnih et al., 2013; Mnih et al., 2015).",
            body,
        )
    )

    story.append(Paragraph("Explicación del código (fragmentos clave)", sh_style))
    code_1 = """# src/preprocesamiento.py

def process_state_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (96, 96))
    return resized.astype(np.float32) / 255.0


def generar_stack_inicial(frame):
    procesado = process_state_image(frame)
    return deque([procesado] * STATE_STACK, maxlen=STATE_STACK)


def actualizar_stack(stack, frame):
    stack.append(process_state_image(frame))
    return stack


def generar_state_frame_stack_from_queue(queue):
    return np.stack(queue, axis=0)
"""
    story.append(Preformatted(code_1, code))

    story.append(PageBreak())

    # ---------------------------------------------------------------------
    # 2) Red neuronal
    # ---------------------------------------------------------------------
    story.append(Paragraph("2) ¿Qué tipo de red neuronal se usa en el DQN?", h_style))

    story.append(Paragraph("Resumen", sh_style))
    story.append(
        Paragraph(
            "Se utiliza una CNN (Convolutional Neural Network) implementada en PyTorch. "
            "La red recibe el estado con forma (4, 96, 96) y produce un vector de 12 Q-valores, "
            "uno por cada acción discretizada.",
            body,
        )
    )

    story.append(Paragraph("¿Dónde está implementada?", sh_style))
    story.append(Paragraph("• Definición de la red: src/agente.py → class DQN(nn.Module)", bullet, bulletText="•"))
    story.append(Paragraph("• Uso de la red en el agente: src/agente.py → class AgenteDQN", bullet, bulletText="•"))

    story.append(Paragraph("Estructura de la arquitectura (alto nivel)", sh_style))
    story.append(Paragraph("• Entrada: 4 canales (stack de frames) × 96 × 96.", bullet, bulletText="•"))
    story.append(Paragraph("• Bloque conv: Conv2d → ReLU → MaxPool → Conv2d → ReLU → MaxPool.", bullet, bulletText="•"))
    story.append(Paragraph("• Bloque denso: Flatten → Linear(flat_dim→216) → ReLU → Linear(216→12).", bullet, bulletText="•"))

    story.append(Paragraph("¿Por qué CNN?", sh_style))
    story.append(
        Paragraph(
            "Una CNN explota la estructura espacial de la imagen (bordes, texturas, formas) y suele ser "
            "la elección estándar para observaciones visuales. En este proyecto la CNN funciona como aproximador de Q(s,a).",
            body,
        )
    )

    story.append(Paragraph("Explicación del código (fragmentos clave)", sh_style))
    code_2 = """# src/agente.py

class DQN(nn.Module):
    def __init__(self, num_acciones):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(STATE_STACK, 6, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 12, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, STATE_STACK, STATE_HEIGHT, STATE_WIDTH)
            conv_out = self.conv(dummy)
            flat_dim = conv_out.numel()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 216),
            nn.ReLU(),
            nn.Linear(216, num_acciones)
        )

    def forward(self, x):
        return self.fc(self.conv(x))
"""
    story.append(Preformatted(code_2, code))

    story.append(PageBreak())

    # ---------------------------------------------------------------------
    # 3) Acción
    # ---------------------------------------------------------------------
    story.append(Paragraph("3) ¿Cuál es mi acción? (steer, gas, brake) y discretización", h_style))

    story.append(Paragraph("Acción original del entorno (continua)", sh_style))
    story.append(
        Paragraph(
            "En CarRacing-v3, la acción original del entorno es un vector continuo de 3 componentes: "
            "(steer, gas, brake). Conceptualmente: steer controla la dirección, gas acelera y brake frena. "
            "En Gymnasium se entrega como un vector numérico (usualmente float32) y el agente debe producir 3 valores continuos.",
            body,
        )
    )

    story.append(Paragraph("¿Qué hace el proyecto? Discretización a 12 acciones", sh_style))
    story.append(
        Paragraph(
            "Como el DQN clásico predice Q-valores para un conjunto finito de acciones, el proyecto discretiza el control "
            "continuo en 12 acciones fijas. Cada acción es una tupla (steer, gas, brake) con valores típicos: steer ∈ {-1,0,1}, "
            "gas ∈ {0,1} y brake ∈ {0,0.2}.",
            body,
        )
    )

    story.append(Paragraph("¿Dónde está implementado?", sh_style))
    story.append(Paragraph("• Definición del ACTION_SPACE: src/entrenamiento.py → ACTION_SPACE", bullet, bulletText="•"))
    story.append(Paragraph("• Misma definición en evaluación: tests/probar_modelo.py → ACTION_SPACE", bullet, bulletText="•"))
    story.append(Paragraph("• Selección de acción por índice: src/agente.py → seleccionar_accion() y memorize()", bullet, bulletText="•"))
    story.append(Paragraph("• Conversión a np.float32 antes de env.step(): src/entrenamiento.py y tests/probar_modelo.py", bullet, bulletText="•"))

    story.append(Paragraph("Tipo de variables y flujo de la acción", sh_style))
    story.append(Paragraph("• En el espacio discreto, una acción es una tupla de Python (steer, gas, brake).", bullet, bulletText="•"))
    story.append(Paragraph("• Al ejecutar en el entorno, se convierte a np.array(..., dtype=np.float32).", bullet, bulletText="•"))
    story.append(Paragraph("• En la memoria, se almacena el índice (int) de la acción para compactar.", bullet, bulletText="•"))

    story.append(Paragraph("Explicación del código (fragmentos clave)", sh_style))
    code_3a = """# src/entrenamiento.py (y tests/probar_modelo.py)

ACTION_SPACE = [
    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),
    (-1, 1, 0), (0, 1, 0), (1, 1, 0),
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
    (-1, 0, 0), (0, 0, 0), (1, 0, 0)
]
"""
    story.append(Preformatted(code_3a, code))

    code_3b = """# tests/probar_modelo.py (ejecución de la acción)

accion = agente.seleccionar_accion(estado)
obs, reward, terminated, truncated, _ = env.step(
    np.array(accion, dtype=np.float32)
)
"""
    story.append(Preformatted(code_3b, code))

    code_3c = """# src/agente.py (memoria compacta)

def memorize(self, state, action, reward, next_state, done):
    action_index = self.action_space.index(action)
    self.memoria.append((state, action_index, reward, next_state, done))
"""
    story.append(Preformatted(code_3c, code))

    story.append(Paragraph("Notas para defensa (qué enfatizar)", sh_style))
    story.append(
        Paragraph(
            "• La discretización es una decisión de ingeniería para aplicar DQN; alternativas para control continuo incluyen SAC/TD3/DDPG o PPO con política continua. "
            "• La tabla de acciones (12) define qué maniobras son posibles; es un hiperparámetro estructural del agente.",
            body,
        )
    )

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
